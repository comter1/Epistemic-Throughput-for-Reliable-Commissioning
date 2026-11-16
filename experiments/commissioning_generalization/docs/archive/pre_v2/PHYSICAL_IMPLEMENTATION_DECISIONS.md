# Physical execution decisions before manifest sealing

The frozen controller equation, gains, physics/control intervals, deadlines, task
amplitudes, failure mechanisms and acceptance tolerances are unchanged. The joint
reference generator draws each waypoint around the Panda home pose with per-joint
uniform amplitudes [0.14,0.10,0.14,0.10,0.12,0.10,0.14] rad. Deadlines are 3.2,6.4,9.6 s.
This task distribution was chosen before any diagnostic models or fault outcomes.

Initial engineering implementations allowed only0.8 s of constant reference before
each deadline. Broader source-only qualification demonstrated insufficient settling
with the mandated state-feedback controller. A uniform1.6 s reference hold still
failed18/48 executions;2.4 s failed3/48 by at most0.000121 m;2.6 s passed96/96
source fixtures with maximum0.019609865 m error. A subsequent exhaustive sweep
of8192 source healthy task seeds found10 further marginal failures (maximum
0.020270098 m), all at the later waypoints. All cases were retained. The final
reference uses uniform0.1 s cubic smoothstep transitions, beginning at0.5,3.2,6.4 s;
this allocates the available interval to physical settling under the mandated
state-feedback servo. The10 previously failing source tasks then pass, and the
full8192-task sweep passed with zero failures and maximum0.019628471 m error
in68.52 s; all756 graph-fixture checks also passed independently. Every
graph receives this same reference timing. No topology, case, fault, method or test
outcome selects this timing. Failed qualification reports are retained under the
external data root; this is a pre-seal engineering timing decision, not evidence of
any acquisition method's quality. Full graph qualification must separately pass.

MuJoCo itself runs5000 steps per10 s execution;1000 executed software ticks route
sensor state through every graph node. Queues use simulation ticks, never sleeps.
Nominal task FK is computed on a separate clean MuJoCo data object. Module repairs
reset only the addressed module's configured delay/gain and transient state.

The public external stream contains measured hand position, nominal reference hand
position, current tracking flag and deployable joint-limit/contact/command-limit
flags at10 Hz. Public verification uses the last six recorded samples in each of
three settled reference plateaus and all public safety flags. This establishes the
declared sampled0.5 s hold criterion at0.1 s resolution. Private audit evaluates the
full100 Hz trajectory; it does not feed the public verifier. Timestamp noise cannot
create a hidden true-time correction. Public verification and continuous audit are
reported as distinct outcomes.

All numeric public payload values are quantized to the actual float32 wire format
before returning decoded observations. Only selected ports are recorded. Recorder
arm/phase streams and per-event timestamp errors are independent of physical sensor
noise and message token randomness. Actual byte charges include default external
records on every additional physical execution and160 B per calibration burst.
