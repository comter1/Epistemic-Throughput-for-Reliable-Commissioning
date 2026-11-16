"""Frozen-reader input checks and explicitly non-deployable record-reading probes.

Run from the experiment directory with PYTHONPATH=.:src. No original data is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import hashlib
import pickle
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
import torch

from cgx.models import features as feature_module
from cgx.models.features import collate_features, tensorize_history
from cgx.provenance import atomic_json, hash_json
from cgx.training.objectives import double_q_target, score_reward
from .core import (inference_settings, load_artifact, load_reader, materialize_prefix,
                   prefix_descriptors, predict, prediction_fields)
from .ledger import Ledger, TimedStage


def relabel_history(history):
    """Rename only identifiers and reorder public node/edge/domain lists."""
    h = deepcopy(history)
    graph = h['graph']
    nodes = {n['id']: f'opaque_node_{1000-i}' for i, n in enumerate(graph['nodes'])}
    clocks = {c: f'opaque_clock_{1000-i}' for i, c in enumerate(graph['clock_domains'])}
    def port(p):
        owner, name = p.split(':', 1)
        return nodes.get(owner, owner) + ':' + name
    for node in graph['nodes']:
        node['id'] = nodes[node['id']]
        node['clock_domain'] = clocks[node['clock_domain']]
    graph['nodes'].reverse()
    graph['clock_domains'] = [clocks[c] for c in reversed(graph['clock_domains'])]
    for edge in graph['edges']:
        edge['source'], edge['target'] = nodes[edge['source']], nodes[edge['target']]
    graph['edges'].reverse()
    for obs in h['observations']:
        for record in obs.get('records', []):
            record['port'] = port(record['port'])
        for burst in obs.get('calibration_bursts', []):
            burst['pair'] = [clocks[c] for c in burst['pair']]
    return h, nodes


def history_metrics(history):
    """Acquired-field coverage; initial common record excluded from paid resources."""
    observations = history['observations'][1:]
    costs = {k: sum(o.get('actual_cost', {}).get(k, 0) for o in observations)
             for k in ('executions', 'calibrations', 'bytes', 'decisions')}
    ports = [p for o in observations for p in sorted({r['port'] for r in o.get('records', [])})]
    actions = [o.get('action_key', '') for o in observations if o.get('records')]
    costs.update(unique_ports=len(set(ports)), repeated_port_visits=len(ports)-len(set(ports)),
                 repeated_actions=len(actions)-len(set(actions)),
                 max_single_record_bytes=max([o.get('actual_cost', {}).get('bytes', 0)
                                              for o in observations]+[0]),
                 raw_records=sum(len(o.get('records', [])) for o in observations))
    return costs


def _fixture():
    graph = json.loads((Path(__file__).resolve().parents[2] / 'examples/public_graph.json').read_text())
    budget = dict(bytes=1000000, executions=4, calibrations=2, decisions=6)
    def record(port, stamp, token='same'):
        return dict(port=port, local_time_s=stamp, message_token=token,
                    event_type='receive', values=[.2]*7, missing=False)
    obs = dict(records=[record('n5:in', 1.), record('n5:out', 1.012)],
               calibration_bursts=[], action_key='fixture', visible_status='ok',
               actual_cost=dict(bytes=256, executions=1, calibrations=0, decisions=1))
    return dict(graph=graph, observations=[obs], remaining=budget.copy(),
                initial_budget=budget.copy(), action_keys=[])


def engineering_checks(ledger):
    """Small analytic inputs test the actual frozen preprocessing implementation."""
    checks = []
    def build(h):
        ledger.reserve('feature_build', detail='input_unit')
        return tensorize_history(h)
    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
    h = _fixture()
    f = build(h)
    check('seconds_and_peer_minus_record_sign',
          np.isclose(f.sequences[0][0,61], .012) and np.isclose(f.sequences[1][0,61], -.012),
          measured_seconds=[float(s[0,61]) for s in f.sequences])
    directions = {n['id']+':'+p['name']: (-1 if p['direction']=='in' else 1)
                  for n in h['graph']['nodes'] for p in n['ports']}
    check('declared_port_direction', all(s[0,53] == directions[p]
          for s,p in zip(f.sequences,['n5:in','n5:out'])))
    split = deepcopy(h)
    split['observations'] = [dict(h['observations'][0], records=[r])
                             for r in h['observations'][0]['records']]
    sf = build(split)
    check('token_equality_is_scoped_to_one_observation',
          all(np.all(s[:,61:63] == 0) for s in sf.sequences))
    repeated = deepcopy(h)
    extra = dict(repeated['observations'][0]['records'][1], local_time_s=1.032)
    repeated['observations'][0]['records'].append(extra)
    rf = build(repeated)
    check('repeated_token_uses_all_acquired_other_port_matches',
          np.isclose(rf.sequences[0][0,61], .022),
          interpretation='All equality matches are pooled; repeated packets are not unique event IDs.')
    appended = deepcopy(h)
    appended['observations'].append(deepcopy(repeated['observations'][0]))
    af = build(appended)
    check('appending_observation_keeps_old_bins_bitwise_equal',
          all(np.array_equal(a,b) for a,b in zip(f.sequences, af.sequences[:len(f.sequences)])))
    large = deepcopy(h)
    large['observations'][0]['records'] = [dict(h['observations'][0]['records'][0],
        local_time_s=i*.001, message_token=f'opaque_{i}', values=[float(i)]*7) for i in range(1000)]
    lf = build(large)
    check('256_bins_keep_last_record_and_cover_full_input',
          len(lf.sequences[0]) == 256 and lf.sequences[0][-1,49] == 999
          and np.isclose(lf.sequences[0][-1,57], .999), bins=len(lf.sequences[0]))
    cal = deepcopy(h)
    a,b = h['graph']['clock_domains'][:2]
    cal['observations'][0]['calibration_bursts'] = [dict(pair=[a,b], timestamps_s=[[1.,1.022,1.023,1.005]])]
    cf = build(cal)
    values = {(int(cf.edge_index[0,i]), int(cf.edge_index[1,i])): float(cf.edge_features[i,1])
              for i,t in enumerate(cf.edge_types) if t == 6}
    check('four_timestamp_sign_seconds_and_reverse_edge',
          np.isclose(values[(cf.clock_lookup[a],cf.clock_lookup[b])], .02)
          and np.isclose(values[(cf.clock_lookup[b],cf.clock_lookup[a])], -.02))
    batch = collate_features([f,lf], 'cpu')
    check('padding_has_exact_sequence_lengths_and_zero_tail',
          batch['lengths'].tolist() == [1,1,256]
          and torch.all(batch['events'][:2,1:] == 0).item()
          and torch.isfinite(batch['events']).all().item())
    reward = torch.tensor([1.,2.,3.,4.])
    online = torch.tensor([[0.,9.,4.]]*4)
    target = torch.tensor([[0.,90.,40.]]*4)
    feasible = torch.tensor([[True,False,True]]*4)
    y = double_q_target(reward,online,target,feasible,torch.tensor([False,True,False,False]),
                        torch.tensor([1,1,0,2]))
    check('double_q_masks_terminal_horizon_and_infeasible_actions',
          torch.equal(y,torch.tensor([41.,2.,3.,44.])), actual=y.tolist())
    lp = [-3.,-1.,-4.,-.5]
    check('score_reward_telescopes_in_bits',
          sum(score_reward(a,b) for a,b in zip(lp,lp[1:])) == lp[-1]-lp[0])
    return checks


class ForwardTimer:
    """Synchronized stage hooks; never mutates the model or feature values."""
    def __init__(self, model):
        self.model, self.rows, self.handles = model, [], []
        self.device = next(model.parameters()).device
    def sync(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
    def __enter__(self):
        for name,module in [('GRU',self.model.backbone.evidence.gru)]+[
                (f'graph_layer_{i}',m) for i,m in enumerate(self.model.backbone.layers)]:
            state={}
            def before(_m,_a,s=state):
                self.sync();s['start']=time.perf_counter()
            def after(_m,_a,_o,s=state,n=name):
                self.sync();self.rows.append((n,time.perf_counter()-s['start']))
            self.handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        self.old_bin=feature_module._bin_sequence
        def timed_bin(*args,**kwargs):
            start=time.perf_counter()
            result=self.old_bin(*args,**kwargs)
            self.rows.append(('binning_including_peer_lookup',time.perf_counter()-start))
            return result
        feature_module._bin_sequence=timed_bin
        return self
    def __exit__(self,*args):
        feature_module._bin_sequence=self.old_bin
        for handle in self.handles:handle.remove()


def run(root, run_dir, device='cuda:0'):
    root,run_dir=Path(root),Path(run_dir)
    ledger=Ledger(run_dir)
    inference_settings()
    checks=engineering_checks(ledger)
    cases=json.loads((run_dir/'manifests/discovery_cases.json').read_text())
    manifest=[dict(case_id=c['case_id'], configuration_id=c['configuration_id'],method=m,policy_seed=11)
              for c in cases for m in ('gnn_depth2','gnn_adaptive')]
    atomic_json(run_dir/'manifests/input_reading_probes.json',dict(
        selection='All 64 preselected discovery cases; both focus methods; seed 11; original terminal acquired history.',
        variants=['original','reverse_observation_blocks','duplicate_first_execution','duplicate_last_execution',
                  'fixed_hash_half_blocks'],jobs=manifest))
    model=load_reader(root,'final',device)
    rows=[]; timing_rows=[]; actual=[]
    with TimedStage(run_dir,'input_checks'):
        for index,job in enumerate(manifest):
            item=load_artifact(root,job['case_id'],job['method'],job['policy_seed'])
            packed,meta=prefix_descriptors(root,item)[-1]
            start=time.perf_counter();history=materialize_prefix(root,packed)
            timing_rows.append(dict(**job,stage='raw_record_read_and_verify',seconds=time.perf_counter()-start,device='cpu'))
            original_hash=hash_json(history)
            obs=history['observations']
            acquired=[i for i,o in enumerate(obs) if i and o.get('records')]
            variants={'original':history,
                      'reverse_observation_blocks':dict(history,observations=list(reversed(obs)))}
            if acquired:
                variants['duplicate_first_execution']=dict(history,observations=obs+[obs[acquired[0]]])
                variants['duplicate_last_execution']=dict(history,observations=obs+[obs[acquired[-1]]])
            else:
                variants['duplicate_first_execution']=history
                variants['duplicate_last_execution']=history
            ranked=sorted(range(1,len(obs)),key=lambda i:hash_json(['input-fixed-half-v1',packed['observation_refs'][i]]))
            kept={0,*ranked[:(len(ranked)+1)//2]}
            variants['fixed_hash_half_blocks']=dict(history,observations=[o for i,o in enumerate(obs) if i in kept])
            baseline=None
            for name,h in variants.items():
                timer=ForwardTimer(model)
                with timer:
                    prediction,timing=predict(model,h,ledger,detail='input_probes')
                p=prediction_fields(prediction)
                if baseline is None:baseline=prediction
                true=item['row']['true_target_key']
                lp=dict(zip(prediction['target_keys'],prediction['log_probabilities_base2']))
                bp=dict(zip(baseline['target_keys'],baseline['log_probabilities_base2']))
                row=dict(**job,variant=name,diagnostic_reader='final_GNN',checkpoint_hash=model.checkpoint_hash,
                         evidence_prefix_hash=meta['evidence_prefix_hash'],true_target_key=true,
                         true_nll_bits=-lp[true], original_timeout=item['row']['timeout'],
                         no_new_measurement=True, original_budgets_unchanged=True,
                         duplicate_applicable=bool(acquired),max_abs_log_probability_change=max(abs(lp[k]-bp[k]) for k in lp),
                         **p,**timing)
                rows.append(row)
                for stage,elapsed in timer.rows:
                    timing_rows.append(dict(**job,variant=name,stage=stage,seconds=elapsed,device=str(device) if stage!='binning_including_peer_lookup' else 'cpu'))
            actual.append(dict(**job,true_target_key=item['row']['true_target_key'],
                               original_false_alarm=item['row']['true_target_key']=='healthy' and item['row']['predicted_target_key']!='healthy',
                               complete_reread_false_alarm=item['row']['true_target_key']=='healthy' and prediction_fields(baseline)['predicted_target_key']!='healthy',
                               **history_metrics(history)))
            if index % 8 == 0:
                # Exact same evidence, changed unused metadata, then renamed IDs/target ordering.
                changed=deepcopy(history)
                changed['remaining']={k:v//2 for k,v in changed['remaining'].items()}
                changed['initial_budget']={k:v*2 for k,v in changed['initial_budget'].items()}
                changed['action_keys']=['unused_metadata_only']
                cp,_=predict(model,changed,ledger,detail='input_metadata')
                delta=max(abs(a-b) for a,b in zip(cp['log_probabilities_base2'],baseline['log_probabilities_base2']))
                checks.append(dict(check='actual_budget_metadata_does_not_enter_diagnoser',**job,passed=delta==0,max_abs_logp_difference=delta))
                renamed,mapping=relabel_history(history)
                rp,_=predict(model,renamed,ledger,detail='input_relabel')
                rpd=dict(zip(rp['target_keys'],rp['log_probabilities_base2']))
                expected={('healthy' if k=='healthy' else mapping[k.split(':')[0]]+':'+k.split(':',1)[1]):v
                          for k,v in zip(baseline['target_keys'],baseline['log_probabilities_base2'])}
                delta=max(abs(expected[k]-rpd[k]) for k in expected)
                checks.append(dict(check='actual_node_clock_relabel_and_target_permutation',**job,passed=delta<1e-4,max_abs_logp_difference=delta))
                ledger.reserve('feature_build',detail='input_padding')
                f=tensorize_history(history)
                ledger.reserve('model_forward',detail='input_padding')
                b=collate_features([f,f],device)
                with torch.inference_mode():lp=model(b).cpu().numpy()
                delta=float(np.max(np.abs(lp[0,:len(f.targets)]-np.asarray(baseline['log_probabilities_base2']))))
                checks.append(dict(check='actual_single_vs_disjoint_batch',**job,passed=delta<1e-4,max_abs_logp_difference=delta,
                                   finite_features=bool(np.isfinite(f.nodes).all() and all(np.isfinite(s).all() for s in f.sequences))))
            if hash_json(history)!=original_hash:raise AssertionError('Input mutated')
            if (index+1)%8==0:print(f'input probes {index+1}/{len(manifest)}',flush=True)
        frame=pd.DataFrame(rows)
        frame.to_parquet(run_dir/'feature_interventions.parquet',index=False)
        frame.drop(columns=['full_log_probabilities','target_keys']).to_csv(run_dir/'feature_interventions.csv',index=False)
        pd.DataFrame(actual).to_csv(run_dir/'input_history_coverage.csv',index=False)
        pd.DataFrame(timing_rows).to_csv(run_dir/'input_runtime_stages.csv',index=False)
        atomic_json(run_dir/'input_checks.json',dict(checks=checks,host=socket.gethostname(),device=device,
            n_failed=sum(not c['passed'] for c in checks),ledger_snapshot=ledger.snapshot()))
        summary=frame.groupby('variant').agg(n=('case_id','size'),mean_logp_change=('max_abs_log_probability_change','mean'),
            max_logp_change=('max_abs_log_probability_change','max')).reset_index()
        summary.to_csv(run_dir/'input_probe_summary.csv',index=False)
        text='''# 输入路径核查（CGX-FA-CAUSE-V1）\n\n原始实现和原数据保持只读。所有比较使用冻结 final GNN checkpoint；完整记录重读没有原 2 秒时限。\n\n'''
        text+=f"执行 {len(checks)} 项工程/真实历史检查，失败 {sum(not c['passed'] for c in checks)} 项。逐项证据见 input_checks.json。\n\n"
        text+='''消息 token 只在单个 observation 内匹配；重复同 token 会匹配同 observation 内所有其他端口的相同 token，因此不能把它解释为唯一全局事件 ID。时间差是 peer.local_time_s − record.local_time_s；校准的正向估计是第二时钟减第一时钟，反向边取负。分箱按每个端口、每次 observation 单独进行，旧块不会因新增 observation 重新分箱；1000 条记录覆盖到末条，压缩为 256 箱。实际有限值、padding 和重排检查均单独记录。\n\n原 Store 公共 history 白名单没有 prediction。DiagnosticReader.predict 调用 tensorize_history(history)，普通路径没有旧 q；PolicyModel.score 显式传入 prediction，训练用 OOF q。tensorize_history 本身允许额外 prediction/planned_actions，所以不可把任意注入字段测试误写成正式路径泄漏。预算被编入 resources，但 DiagnosticReader/Backbone 不消费该数组，动作 head 才读取。真实 history 的预算与 action_keys 改动验证了这一点。\n\n反转 observation 块、复制已取得执行块、固定 hash 保留一半块是解释性读法干预：没有新增物理测量，复制会改变 log-count 和端口序列平均权重，因此预测改变不是自动的代码错误。初始块也参与反转，因为模型采用分块编码后池化；块内事件顺序从不改变。half-blocks 会删去证据，只能解释读法敏感性，不能当成新部署策略或同成本性能胜利。所有原始预算保持原值，逐案例带原 timeout。\n\n计时使用远端 B200；raw read 与 feature 工作在 CPU。binning_including_peer_lookup 同时包含相等 token 查找和分箱，不能把它当作纯分箱时间；计时 hook 同步 GPU，不可用这些秒数替代原 CPU 单线程 deadline 统计。\n\n'''
        text+=summary.to_string(index=False)+'\n'
        (run_dir/'input_path_checks.md').write_text(text)


def additional_checks(root,run_dir):
    """Physical calibration is observation-only; inspect the ordinary q input path."""
    from cgx.env.system import System
    root,run_dir=Path(root),Path(run_dir);ledger=Ledger(run_dir)
    cases=json.loads((run_dir/'manifests/discovery_cases.json').read_text())
    case=cases[0]
    item=load_artifact(root,case['case_id'],'gnn_depth2',11)
    packed,_=prefix_descriptors(root,item)[-1]
    history=materialize_prefix(root,packed)
    checks=[]
    with TimedStage(run_dir,'additional_input_checks'):
        system=System(history['graph'],case,history['nominal_task'],asset_path=root/'assets/mujoco_menagerie/franka_emika_panda')
        def state_digest():
            # Internal state is used only by this evaluator assertion, never as model input.
            return hashlib.sha256(pickle.dumps((system.offsets,system.modules,system.data.qpos.copy(),
                system.data.qvel.copy(),system.data.time),protocol=5)).hexdigest()
        before=state_digest()
        ledger.reserve('calibration',detail='input_calibration_state')
        result=system.calibrate(history['graph']['clock_domains'][:2],burst_index=0)
        after=state_digest()
        checks.append(dict(check='System.calibrate_does_not_change_clock_offsets_modules_or_physics',passed=before==after,
                           before_state_hash=before,after_state_hash=after,actual_cost=result.observation['actual_cost'],
                           observation_hash=hash_json(result.observation),case_id=case['case_id']))
        ledger.reserve('feature_build',detail='input_q_contract')
        features=tensorize_history(history)
        checks.append(dict(check='stored_history_has_no_previous_q_and_ordinary_q_channels_zero',
                           passed='prediction' not in history and bool(np.all(features.nodes[:,27:29]==0)),
                           public_history_keys=sorted(history)))
        injected=dict(history,prediction=dict(target_keys=['healthy'],log_probabilities_base2=[0.]))
        ledger.reserve('feature_build',detail='input_q_contract')
        extra=tensorize_history(injected)
        checks.append(dict(check='explicit_noncontract_q_is_a_real_selector_input',
                           passed=bool(np.all(extra.nodes[:,28]==1)),
                           interpretation='Injected prediction is outside Store public-history whitelist; ordinary reader never receives it.'))
        records=[r for o in history['observations'] for r in o.get('records',[])]
        finite=all(np.isfinite([r['local_time_s'],*r['values']]).all() for r in records)
        checks.append(dict(check='sampled_actual_raw_record_values_and_timestamps_finite',passed=finite,records=len(records)))
        same_token_other_port=0;ambiguous=0
        for obs in history['observations']:
            matches=defaultdict(list)
            for r in obs.get('records',[]):matches[r['message_token']].append(r)
            for rs in matches.values():
                counts=Counter(r['port'] for r in rs)
                for r in rs:
                    if len(counts)>1:
                        same_token_other_port+=1
                        ambiguous+=any(n>1 for p,n in counts.items() if p!=r['port'])
        checks.append(dict(check='actual_repeated_token_ambiguity_documented',passed=True,
                           matched_records=same_token_other_port,records_with_repeated_peer_port=ambiguous,
                           interpretation='Observed equality ambiguity is counted; no unique event association is invented.'))
    atomic_json(run_dir/'additional_input_checks.json',dict(checks=checks,n_failed=sum(not c['passed'] for c in checks),
                                                           ledger_snapshot=ledger.snapshot()))
    print(json.dumps(checks,ensure_ascii=False),flush=True)


def summarize_results(run_dir):
    run_dir=Path(run_dir)
    frame=pd.read_csv(run_dir/'feature_interventions.csv');rows=[]
    for (method,variant),group in frame.groupby(['method','variant']):
        healthy=group[group.true_target_key=='healthy'];fault=group[group.true_target_key!='healthy']
        rows.append(dict(method=method,variant=variant,n=len(group),healthy_n=len(healthy),fault_n=len(fault),
                         healthy_false_alarms=int((healthy.predicted_target_key!='healthy').sum()),
                         fault_misses=int((fault.predicted_target_key=='healthy').sum()),
                         mean_nll=group.true_nll_bits.mean(),max_logp_change=group.max_abs_log_probability_change.max()))
    summary=pd.DataFrame(rows)
    summary.to_csv(run_dir/'input_probe_outcome_summary.csv',index=False)
    text='\n\n补充：完成后的逐方法读法敏感性\n\n'
    text+='每方法64个固定发现案例（32健康/32故障），仅seed11。以下完整记录基线已移除原deadline；重复/半块结果不能与原返回旧预测混为一谈。复制对象是整个已取得execution observation，若它包含真实测得的calibration burst，该burst也随块复制；没有测量新数据。\n\n'
    text+=summary.to_string(index=False)+'\n\n'
    text+='块顺序反转只有浮点舍入级差异；复制首/末记录会改变序列池化权重和count特征。健康误报下降时同时存在故障漏检上升，因此这既不是修复证明，也不是部署性能改进。\n\n'
    text+='additional_input_checks.json记录实际System.calibrate调用前后的offset/module/physics状态hash一致，以及真实公共history没有旧q。cpu_gpu_batch_parity.csv提供另一个agent的异构graph/target数padding核查；intervention_runtime_stage_breakdown.csv提供单独message-matching微基准，其含义与端到端feature计时不同、存在重叠，不能直接相加。\n'
    path=run_dir/'input_path_checks.md';original=path.read_text().split('\n\n补充：完成后的逐方法读法敏感性')[0]
    path.write_text(original+text)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--additional-checks',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:summarize_results(args.run_dir)
    elif args.additional_checks:additional_checks(args.data_root,args.run_dir)
    else:run(args.data_root,args.run_dir,args.device)
