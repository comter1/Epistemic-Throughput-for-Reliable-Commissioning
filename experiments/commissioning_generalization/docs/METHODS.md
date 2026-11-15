# Evidence collection methods and the shared GNN diagnoser

The six main methods differ in how they choose and collect evidence. After evidence
collection, they give their acquired records to the same final GNN diagnoser, with
the same frozen weights. The main comparison therefore measures differences in
evidence collection and its execution under the declared resource limits, rather
than differences between final diagnostic models. Timeouts remain part of that
comparison; a timed-out method retains its last valid prediction and abstains.

The display names below describe the existing CGX-2.0 implementation. Internal
method keys, checkpoints, protocol settings and experimental results are unchanged.
The centralized mapping for report and figure generation is
[`src/cgx/reporting/methods.py`](../src/cgx/reporting/methods.py).

| Internal key | Full display name |
|---|---|
| `gnn_fixed_plan` | **GNN Plan-Once + GNN Diagnoser** |
| `gnn_adaptive` | **GNN Replan-Each-Round + GNN Diagnoser** |
| `transformer_adaptive` | **Transformer Replan-Each-Round + GNN Diagnoser** |
| `gnn_depth2` | **GNN 2-Step Planner + GNN Diagnoser** |
| `cost_aware_adaptation` | **GNN Cost-Aware One-Step + GNN Diagnoser** |
| `full_recording_feasible` | **Full Recording + GNN Diagnoser** |

Compact figure legends may omit `+ GNN Diagnoser` only when the caption explicitly
states that all main methods use the same final GNN diagnoser. The full names belong
in the main method table and explanatory text. An evidence-collection column and a
separate final-diagnoser column may also make the two roles explicit.

## GNN Plan-Once + GNN Diagnoser

A GNN trained on source systems reads the new system's public structure, initial
evidence and available diagnostic budget. It creates the complete inspection plan
for that particular system before collecting new diagnostic evidence. The plan can
therefore differ between systems and initial observations; it is not a hand-written
checklist shared by every system.

The system then executes the planned inspections without using intermediate
observations to revise later steps. Once the planned observations have been
collected, the shared GNN diagnoser predicts either healthy operation or the faulty
component and fault type. The word "Once" refers to when the inspection plan is
made, not to the number of inspections or executions in that plan.

## GNN Replan-Each-Round + GNN Diagnoser

A trained GNN chooses one inspection from the currently available options. After
that inspection returns new evidence, the method reads the updated investigation
history and chooses the next inspection again. Later choices can change in response
to earlier observations. This continues until the method stops or reaches a
resource limit. The shared GNN diagnoser makes the final fault prediction.

## Transformer Replan-Each-Round + GNN Diagnoser

A trained Transformer chooses one inspection at a time, reads each new observation,
and chooses the next inspection from the updated history. Its role is evidence
selection. Final diagnosis is still made by the same shared GNN diagnoser used by
the other main methods.

The Transformer receives the same public system connections, acquired evidence,
available inspections and resource limits as the GNN evidence selector. Its
attention layers include the public relationship information. Thus this comparison
changes the model that chooses inspections, not the model that makes the final
diagnosis or whether the selector can see system connections.

## GNN 2-Step Planner + GNN Diagnoser

A trained GNN chooses the next inspection while considering what one additional
inspection could be useful afterward. After receiving new evidence, it chooses
again with the updated history and remaining resources. Its learned value model
uses one or two remaining decisions, depending on what the remaining budget allows.

"2-Step" describes this planning horizon. It does not describe the number of
neural-network layers, and it does not mean that the entire investigation is limited
to two executions. The value estimates are learned from source systems; the method
does not query hidden test-system outcomes to perform exact planning. The shared
GNN diagnoser produces the final prediction.

## GNN Cost-Aware One-Step + GNN Diagnoser

A trained GNN estimates how useful each currently available inspection would be
for diagnosis. The selection rule combines the predicted next diagnostic loss with
the inspection's declared cost in recorded bytes, executions and calibrations.
The balance between predicted diagnostic quality and cost is selected on validation
systems and frozen before testing.

The method chooses the next inspection, reads its outcome, and then compares the
available inspections again. It evaluates the immediate next inspection rather
than assigning value to a complete sequence of future observations. The choice
uses a learned GNN, not a hand-written cost rule alone. Final diagnosis uses the
shared GNN diagnoser. The internal training key `gnn_myopic` supplies this model;
the deployed method retains the internal key `cost_aware_adaptation`.

## Full Recording + GNN Diagnoser

This method follows the full-recording rule and records the full allowed set of
signals that fits its reference condition. It does not use a learned model to
choose what evidence to record. The resulting records are passed to the same
learned GNN diagnoser used by the other main methods.

In the main matrix this reference is registered only at B=4, with its actual
recording and execution costs reported. Calibration options remain available under
the same rules; observed calibration usage must be read from the saved results.
Comparisons with other methods concern their complete acquisition plans, including
which signals and calibration measurements were collected.

## What the shared GNN diagnoser does

The implementation in [`models/networks.py`](../src/cgx/models/networks.py) first
maps each 64-value event-summary token to a 32-dimensional embedding. A GRU with
64 hidden units reads each recorded interface sequence. The sequence summaries
are attached to their corresponding system components, together with public
component information and the external observations. A GNN of width 128 passes
information along the system connections through three message-passing layers.
The output is a normalized probability distribution over `healthy` and each legal
`(component, fault type)` target for that system.

The diagnosis model is separate from the inspection-selection models, even when
both use a GNN. It is shared across all six main methods and frozen during their
evaluation. Three source-system cross-fit readers provide training scores without
having trained on the topology being scored; these training readers are also
distinct from the shared final diagnoser.

The independent Relation-Transformer reader is a separate diagnostic model used
only to re-read the already frozen transcripts. It checks how another model uses
exactly the same collected evidence, without acquiring new observations. It is not
the main GNN diagnoser, and it is not the Transformer evidence selector in
**Transformer Replan-Each-Round + GNN Diagnoser**.

## Historical names and reproducibility

Original generated reports and figures, archived documents, and sealed source
snapshots retain their legacy display names for provenance. Names such as
"Fixed plan", "GNN", "Relation Transformer", "Short-horizon planner" and
"Cost-aware myopic" in those historical artifacts refer to the internal keys in
the table above. They should not be reused as standalone names in new main reports.
The legacy "Fixed plan" label refers to GNN Plan-Once, which learns to create a
system-specific plan. The legacy "Full recording" label still includes a learned
final GNN diagnoser.

Corrected presentation exports are separate from the original result files. CSV
exports retain their internal `method` column and add a `display_name` column.
Renaming presentation artifacts does not change training seeds, model parameters,
checkpoint selection, protocol hashes or raw experimental measurements.
