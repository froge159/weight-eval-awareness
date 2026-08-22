"""evalaware -- Phase 1 (step 1) of weight-space removal of evaluation awareness.

Scope: this package builds the **contrastive matrix** X = H_eval - H_deploy.
Step 2 (SVD / direction isolation) and step 3 (sanity checks) are owned by
other group members. See HANDOFF.md for the artifact contract.

Modules
-------
validation    Gates used everywhere. Every check names expected vs observed.
prompts       Prompt inventory, transcribed from Hua et al. and verified
              byte-for-byte against their released JSON.
reference     Validates our transcription against the authors' repo data.
dataset       Matched pair construction and disjoint splitting.
activations   Residual-stream capture (needs torch; imported lazily).
contrastive   ** Step 1 deliverable. ** Builds, validates and saves X.
behavior      Type-hint scorer and baseline-rate gate.
config        Config loading, seeding, run provenance.

extraction    Step 2 REFERENCE ONLY -- not owned by this workstream.
checks/       Step 3 REFERENCE ONLY -- not owned by this workstream.
"""

__version__ = "0.1.0"

from . import contrastive, dataset, prompts, validation  # noqa: F401

__all__ = ["contrastive", "dataset", "prompts", "validation", "__version__"]
