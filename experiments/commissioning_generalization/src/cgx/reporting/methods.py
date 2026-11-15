"""Public method names; scientific identifiers and model choices stay unchanged."""

# Retain the original plotting order so each method keeps its existing color.
COMPACT_NAMES = {
    "gnn_fixed_plan": "GNN Plan-Once",
    "cost_aware_adaptation": "GNN Cost-Aware One-Step",
    "gnn_depth2": "GNN 2-Step Planner",
    "transformer_adaptive": "Transformer Replan-Each-Round",
    "gnn_adaptive": "GNN Replan-Each-Round",
    "full_recording_feasible": "Full Recording",
}
DISPLAY_NAMES = {key: name + " + GNN Diagnoser" for key, name in COMPACT_NAMES.items()}
SHARED_DIAGNOSER_CAPTION = (
    "All methods use the same final GNN diagnoser; labels describe how evidence is collected."
)
METHOD_DESCRIPTIONS = {
    "gnn_fixed_plan": (
        "A trained GNN reads the new system structure, initial evidence and available budget, "
        "then creates a complete inspection plan for that system before collecting new "
        "diagnostic evidence. Later inspections do not change in response to intermediate "
        "observations. The shared GNN diagnoser reads the collected records and makes the final diagnosis."
    ),
    "gnn_adaptive": (
        "A trained GNN chooses one inspection, reads the updated evidence, then chooses "
        "the next inspection again. Its choices can change after every round until it "
        "stops or exhausts its budget. The shared GNN diagnoser makes the final diagnosis."
    ),
    "transformer_adaptive": (
        "A trained Transformer chooses one inspection at a time from the public system "
        "connections, acquired evidence and remaining budget. After each new observation "
        "it chooses again. The Transformer selects evidence; the same shared GNN diagnoser "
        "used by the other main methods makes the final diagnosis."
    ),
    "gnn_depth2": (
        "A trained GNN chooses the next inspection using a learned value that considers "
        "that inspection and up to one following inspection. It chooses again when new "
        "evidence arrives. Two-step refers to this planning horizon, not the number of GNN "
        "layers. The shared GNN diagnoser makes the final diagnosis."
    ),
    "cost_aware_adaptation": (
        "A trained GNN estimates how useful each currently available inspection would be "
        "for diagnosis. The method combines that estimate with the inspection cost to "
        "choose the next inspection, then repeats after receiving new evidence. It evaluates "
        "the immediate next inspection rather than a complete multiround sequence. "
        "The shared GNN diagnoser makes the final diagnosis."
    ),
    "full_recording_feasible": (
        "A full-recording rule collects the allowed signals that fit the reference "
        "condition. Evidence selection is not learned. The records are passed to the "
        "same learned GNN diagnoser used by all other main methods."
    ),
}


def display_name(key, compact=False):
    """Return a public name without changing unknown or historical identifiers."""
    return (COMPACT_NAMES if compact else DISPLAY_NAMES).get(key, key)
