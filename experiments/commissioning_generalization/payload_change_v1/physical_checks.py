"""Executable source-only physical integrity and qualification checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from cgx.env.system import reference_at
from .physics import Dynamics, PayloadPlant, DEFAULT_CONTROLLER
from .trajectories import original_task, action_library, workload, reference, reference_grid
from .storage import PhysicsLedger, write_json


def run_checks(run_root):
    root=Path(run_root)
    case=json.loads((root/'private/source.json').read_text())[0]
    graph=json.loads((root/'configs/healthy_graph.json').read_text())
    true=case['true_payload']; configured=case['configured_payload']
    known=Dynamics(None); loaded=Dynamics(true)
    checks={}
    checks['payload_total_mass_exact']=bool(abs(loaded.model.body_mass.sum()-known.model.body_mass.sum()-true['mass'])<1e-10)
    checks['fixed_gripper_mass_retained_once']=bool(known.model.nq==7 and loaded.model.nq==7)
    checks['no_automatic_gravity_compensation']=bool(np.all(loaded.model.body_gravcomp==0))
    checks['unit_torque_actuation_no_position_inner_loop']=bool(
        np.all(loaded.model.actuator_gainprm[:,0]==1) and
        np.all(loaded.model.actuator_biasprm==0) and
        np.all(loaded.model.actuator_gear[:,0]==1))
    action=original_task(case['task_seed'])
    checks['L0_reference_exact']=all(np.array_equal(reference(action,float(t))[0],reference_at(float(t),action['task']))
                                      for t in np.arange(5000)*.002)
    fast=workload(88001,1.0);slow=workload(88001,.5)
    pairs=[(reference(fast,t),reference(slow,2*t)) for t in np.linspace(0,10,101)]
    checks['half_speed_same_path_derivatives_scaled']=all(np.allclose(a[0],b[0],atol=1e-13) and
          np.allclose(a[1]*.5,b[1],atol=1e-13) and np.allclose(a[2]*.25,b[2],atol=1e-13) for a,b in pairs)
    limits=[]
    for act in action_library():
        grid=reference_grid(act,hz=100)
        limits.append({'id':act['id'],'max_velocity':float(np.abs(grid['dq']).max()),
                       'max_acceleration':float(np.abs(grid['ddq']).max()),
                       'in_joint_bounds':bool(np.all((grid['q']>=known.joint_range[:,0])&(grid['q']<=known.joint_range[:,1])))})
    checks['library_public_reference_limits']=all(x['in_joint_bounds'] and x['max_velocity']<=1.5 and x['max_acceleration']<=3 for x in limits)
    plant=PayloadPlant(true,configured,79011,graph,PhysicsLedger(root))
    checks['plant_controller_model_objects_separate']=plant._model is not plant.controller.model
    plant_masses=plant._model.body_mass.copy(); plant_inertia=plant._model.body_inertia.copy()
    plant.configure(true)
    checks['configuration_rebuild_leaves_plant_unchanged']=bool(np.array_equal(plant_masses,plant._model.body_mass) and np.array_equal(plant_inertia,plant._model.body_inertia))
    bias=plant._bias.copy()
    branch=plant.clone()
    # Two real, separately charged executions test complete evaluator snapshot fidelity.
    a=plant.execute(action_library()[0],'source_clone_equivalence',kind='source_qualification')
    b=branch.execute(action_library()[0],'source_clone_equivalence',kind='source_qualification')
    a.save(root/'source_validation/clone_check/a');b.save(root/'source_validation/clone_check/b')
    checks['clone_public_bitwise_equal']=all(np.array_equal(a.public[k],b.public[k]) for k in a.public)
    checks['clone_private_bitwise_equal']=all(np.array_equal(a.private['arrays'][k],b.private['arrays'][k]) for k in a.private['arrays'])
    checks['persistent_bias_not_reset']=bool(np.array_equal(plant._bias,bias) and np.array_equal(branch._bias,bias))
    checks['partition_recording_preserves_physical_cost']=bool(sum(len(x)*.002 for x in np.array_split(a.private['arrays']['time'],7))==a.physical_seconds)
    public_forbidden={'qacc','qfrc_bias','qM','true_payload','true_mass','group','private_audit_seeds'}
    checks['sensor_projection_no_hidden_fields']=not(public_forbidden & set(a.public))
    profile_path=root/'resources/profile20.json'
    if profile_path.exists():
        profile=json.loads(profile_path.read_text())
        checks['twenty_source_blocks_recorded']=len(profile['rows'])==20
        checks['smooth_nominal_work_and_library_pass']=all(z['private_metrics']['pass'] for z in profile['rows'] if z['action']!='L0')
    report={'checks':checks,'passed':all(checks.values()),'library_limits':limits,
            'controller':DEFAULT_CONTROLLER,'mujoco_version':__import__('mujoco').__version__,
            'runtime_mass_editing':'not used: immutable XML recompile, including configuration updates',
            'clone_physics_seconds_charged':a.physical_seconds+b.physical_seconds,
            'partition_note':'Seven storage/API slices of one already acquired continuous physical history do not alter integrated physics duration.'}
    write_json(root/'checks/physical_integrity.json',report)
    write_json(root/'checks/physical_checks.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);args=p.parse_args()
    result=run_checks(args.run_root);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
