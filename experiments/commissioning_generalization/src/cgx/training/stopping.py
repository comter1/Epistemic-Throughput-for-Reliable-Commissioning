"""Predeclared validation-only stopping; checkpoint selection remains the true minimum."""
import math


def should_stop(history,training):
    rule=training.get('early_stopping')
    if not rule or not history:return False
    anchor=math.inf;stale=0
    for row in history:
        value=row['mean_final_nll_bits']
        if not math.isfinite(value):raise ValueError('Nonfinite validation metric')
        if value<anchor-rule['min_delta_nll_bits']:anchor=value;stale=0
        else:stale+=1
    return stale>=rule['patience_checks']


def validate_completion(completion,history,training,cap):
    updates=completion.get('actual_updates',0)
    if not history or history[-1]['iteration']!=updates or not 0<updates<=cap:raise ValueError('Training completion/update history mismatch')
    interval=training.get('validation_interval')
    if interval:
        expected=list(range(interval,updates+1,interval))
        if updates==cap and updates%interval:expected.append(updates)
        if [r['iteration'] for r in history]!=expected:raise ValueError('Missing/duplicate/out-of-order validation checks')
    if any(not math.isfinite(r['mean_final_nll_bits']) for r in history):raise ValueError('Nonfinite validation history')
    if updates<cap:
        if completion.get('termination_reason')!='validation_patience' or not should_stop(history,training):
            raise ValueError('Short training lacks the registered validation-only early stop')
        if completion.get('registered_update_cap')!=cap or completion.get('early_stopping_rule')!=training.get('early_stopping'):
            raise ValueError('Early-stop rule differs from the registered configuration')
    return True
