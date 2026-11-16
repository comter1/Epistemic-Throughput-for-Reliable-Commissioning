"""One numerical authority for the registered method/seed/resource matrix."""
def main_methods(config):
    if config.get('protocol_id')=='CGX-2.0':
        return [(m,s) for m in config['methods']['main_learned_evaluation'] for s in config['seeds']['training']]+[(m,None) for m in config['methods']['nonlearned']]
    return ([(m,s) for m in config['methods']['learned'] for s in config['seeds']['training']]
            +[(m,None) for m in config['methods']['nonlearned']]
            +[('cost_aware_adaptation',s) for s in config['seeds']['training']])


def conditions(config,method,sensitivity=False):
    if sensitivity:caps=config['resources']['round_sensitivity_execution_caps']
    elif config.get('protocol_id')=='CGX-2.0':
        caps=[config['resources']['full_recording_execution_cap']] if method.startswith('full_recording') else config['resources']['main_execution_budgets']
    else:caps=[config['evaluation']['main_execution_cap']]
    return [(cap,fraction) for cap in caps for fraction in config['resources']['byte_fractions']]


def matrix_cells(config,sensitivity=False):
    return [(method,seed,cap,fraction) for method,seed in main_methods(config) for cap,fraction in conditions(config,method,sensitivity)]


def trained_method(method):
    return 'gnn_myopic' if method=='cost_aware_adaptation' else method


def seeded_method(method,config):
    return method in config['methods']['learned'] or method=='cost_aware_adaptation'
