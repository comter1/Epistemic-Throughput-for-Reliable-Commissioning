"""Fail closed if frozen numeric assumptions change without implementation amendment."""
from ..config import load_config

def require_supported(config=None):
    config = load_config() if config is None else config
    expected = {
        'system': {'physics_dt_seconds':.002,'control_hz':100,'execution_seconds':10.,'waypoints':3,
                   'position_tolerance_m':.02,'hold_seconds':.5,'joint_limit_tolerance_rad':.001,
                   'servo_alpha':.2,'servo_velocity_damping_seconds':.02},
        'sensing': {'joint_position_std_rad':.0001,'joint_velocity_std_rad_s':.001,
                    'timestamp_independent_std_ms':1.,'calibration_burst_exchanges':4,
                    'calibration_link_one_way_mean_ms':2.,'calibration_link_one_way_std_ms':.5,
                    'calibration_processing_delay_ms':.1,'recorder_arm_delay_uniform_ms':[0.,100.],
                    'physical_clock_adjustment':False,'perfect_shared_noise_in_main':False},
        'observations': {'sample_hz':[10,50,100],'default_external_hz':10,'trace_record_bytes':128,
                         'calibration_burst_bytes':160,'calibrations_per_action_max':1},
        'resources': {'reference_executions_for_byte_cap':4,'main_execution_cap':4,
                      'main_calibration_cap':2,'main_decision_cap':6},
    }
    for section, values in expected.items():
        for key, value in values.items():
            if config[section][key] != value:
                raise ValueError(f'Unsupported protocol amendment {section}.{key}: {config[section][key]!r}')
    return config
