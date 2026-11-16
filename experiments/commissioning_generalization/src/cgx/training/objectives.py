"""Training-only objectives: private labels never cross the selector interface."""
import torch


def joint_nll_bits(log_probabilities_base2,labels):
    return -log_probabilities_base2.gather(1,labels[:,None]).squeeze(1).mean()


def score_reward(before_log2,after_log2):
    return after_log2-before_log2


def double_q_target(reward,online_next,target_next,feasible,terminal,remaining_decisions):
    """Undiscounted finite-horizon target; STOP must be a feasible zero column."""
    chosen=online_next.masked_fill(~feasible,-torch.inf).argmax(-1)
    continuation=target_next.gather(1,chosen[:,None]).squeeze(1)
    return reward+torch.where(terminal | (remaining_decisions<=0),torch.zeros_like(continuation),continuation)


def frozen(model):
    model.eval()
    for module in model.modules():
        if isinstance(module,torch.nn.RNNBase):module.flatten_parameters()
    for p in model.parameters(): p.requires_grad_(False)
    return model
