"""Permutation-equivariant diagnostic and finite-horizon acquisition networks."""
from __future__ import annotations
import math
from contextlib import nullcontext
import torch
from torch import nn
from .features import NODE_WIDTH, ACTION_WIDTH, tensorize_history, collate_features, collate_actions


def mean_pool(x,index,n):
    out=x.new_zeros((n,x.shape[-1])); out.index_add_(0,index,x)
    counts=torch.bincount(index,minlength=n).clamp_min(1).to(x.dtype)
    return out/counts[:,None]


class EvidenceEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.event=nn.Linear(64,32); self.gru=nn.GRU(32,64,batch_first=True)
    def forward(self,b):
        # CuDNN GRU is kept FP32; graph/attention layers use the configured BF16 autocast.
        ctx=torch.autocast(device_type=b['events'].device.type,enabled=False)
        with ctx:
            x=torch.tanh(self.event(b['events'].float()))
            packed=nn.utils.rnn.pack_padded_sequence(x,b['lengths'],batch_first=True,enforce_sorted=False)
            _,h=self.gru(packed); h=h[-1]
        owner=b['sequence_owners']; observed=(owner>=0)&b['sequence_valid']; external_mask=(owner<0)&b['sequence_valid']
        local=mean_pool(h[observed],owner[observed],len(b['nodes']))
        external=mean_pool(h[external_mask],b['sequence_batch'][external_mask],b['batch_size'])
        return local,external


class MessageLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.edge=nn.Embedding(7,32)
        self.message=nn.Sequential(nn.Linear(width+36,width),nn.SiLU(),nn.Linear(width,width))
        self.update=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)
    def forward(self,x,b):
        source,target=b['edge_index']
        msg=self.message(torch.cat((x[source],self.edge(b['edge_types']),b['edge_features']),-1))
        aggregate=mean_pool(msg,target,len(x))
        return self.norm(x+self.update(torch.cat((x,aggregate),-1)))


class RelationAttention(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__(); self.heads=heads; self.width=width
        self.qkv=nn.Linear(width,width*3); self.projection=nn.Linear(width,width)
        self.relation_bias=nn.Embedding(8,heads); self.attribute_bias=nn.Linear(4,heads,bias=False)
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
    def forward(self,x,b):
        B=b['batch_size']; N=max(b['sizes']); H=self.heads; D=self.width//H
        padded=x.new_zeros(B,N,self.width); present=torch.zeros(B,N,dtype=torch.bool,device=x.device)
        for i,(o,s) in enumerate(zip(b['offsets'],b['sizes'])): padded[i,:s]=x[o:o+s]; present[i,:s]=True
        q,k,v=self.qkv(padded).reshape(B,N,3,H,D).permute(2,0,3,1,4).unbind(0)
        bias=self.relation_bias.weight[7].reshape(1,H,1,1).expand(B,H,N,N).clone()
        src,dst=b['edge_index']; node_batch=b['node_batch']; offsets=torch.as_tensor(b['offsets'],device=x.device)
        groups=node_batch[src]; ls=src-offsets[groups]; ld=dst-offsets[groups]
        relation=self.relation_bias(b['edge_types'])+self.attribute_bias(b['edge_features'])
        bias[groups,:,ld,ls]=relation
        attention=(q@k.transpose(-2,-1))/math.sqrt(D)+bias
        attention=attention.float().masked_fill(~present[:,None,None,:],-torch.inf).softmax(-1).to(v.dtype)
        z=(attention@v).transpose(1,2).reshape(B,N,self.width)
        z=self.norm1(padded+self.projection(z)); z=self.norm2(z+self.ff(z))
        return torch.cat([z[i,:s] for i,s in enumerate(b['sizes'])])


class Backbone(nn.Module):
    def __init__(self,architecture='gnn',width=128):
        super().__init__(); self.architecture=architecture; self.width=width
        self.evidence=EvidenceEncoder(); self.input=nn.Linear(NODE_WIDTH+128+34,width)
        self.layers=nn.ModuleList([MessageLayer(width) if architecture=='gnn' else RelationAttention(width) for _ in range(3)])
    def forward(self,b):
        local,external=self.evidence(b)
        x=torch.tanh(self.input(torch.cat((b['nodes'],local,external[b['node_batch']],b['task'][b['node_batch']]),-1)))
        for layer in self.layers: x=layer(x,b)
        pooled=mean_pool(x,b['node_batch'],b['batch_size'])
        global_summary=torch.cat((pooled,torch.log1p(b['physical_counts'].float())[:,None]),-1)
        return x,global_summary


class DiagnosticReader(nn.Module):
    def __init__(self,architecture='gnn',width=128,floor=1e-8):
        super().__init__(); self.architecture=architecture; self.width=width; self.floor=floor
        self.backbone=Backbone(architecture,width)
        self.family=nn.Embedding(4,16)
        self.target_head=nn.Sequential(nn.Linear(width*2+17,width),nn.SiLU(),nn.Linear(width,1))
        self.healthy_head=nn.Sequential(nn.Linear(width+1,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self,b):
        nodes,glob=self.backbone(b); ti=b['target_nodes']; safe=ti.clamp_min(0)
        target=self.target_head(torch.cat((nodes[safe],glob[:,None,:].expand(-1,ti.shape[1],-1),self.family(b['target_families'])),-1)).squeeze(-1)
        target=torch.where(ti<0,self.healthy_head(glob),target)
        logits=target.float().masked_fill(~b['target_valid'],-torch.inf)
        logp=torch.log_softmax(logits,-1)
        # Apply floor and normalization identically in training, scoring and inference.
        p=logp.exp().clamp_min(self.floor)*b['target_valid']
        p=p/p.sum(-1,keepdim=True)
        return p.clamp_min(self.floor).log()/math.log(2)
    @torch.inference_mode()
    def predict(self,history):
        self.eval(); device=next(self.parameters()).device
        f=tensorize_history(history); b=collate_features([f],device)
        lp=self(b)[0,:len(f.targets)].cpu().tolist()
        return {'target_keys':list(f.targets),'log_probabilities_base2':lp}


class PolicyModel(nn.Module):
    def __init__(self,method='gnn_adaptive',width=128):
        super().__init__(); self.method=method; self.width=width
        arch='transformer' if method=='transformer_adaptive' else 'gnn'
        self.backbone=Backbone(arch,width)
        self.action_head=nn.Sequential(nn.Linear(width*2+1+ACTION_WIDTH,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        self.next_loss_head=nn.Sequential(nn.Linear(width*2+1+ACTION_WIDTH,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,b,actions,return_next_loss=False):
        x,g=self.backbone(b)
        p=mean_pool(x[actions['owners']],actions['pool_action'],len(actions['features']))
        inp=torch.cat((p,g[actions['batch']],actions['features']),-1)
        score=self.action_head(inp).squeeze(-1).float()
        # STOP is exactly zero rather than a fitted approximation to terminal value.
        score=score.masked_fill(actions['stop'],0.)
        if return_next_loss: return score,self.next_loss_head(inp).squeeze(-1).float()
        return score
    @torch.inference_mode()
    def score(self,history,actions,prediction=None,return_next_loss=False):
        self.eval(); device=next(self.parameters()).device
        f=tensorize_history(history,prediction=prediction); b=collate_features([f],device)
        a=collate_actions(b,[actions],device)
        return self(b,a,return_next_loss=return_next_loss)


def parameter_count(model): return sum(p.numel() for p in model.parameters())
