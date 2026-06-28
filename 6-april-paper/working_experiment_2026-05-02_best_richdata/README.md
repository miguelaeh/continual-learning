# Preserved Snapshot: 2026-05-02 Best Rich-Data Run

This snapshot preserves the exact code and checkpoints used for the current best result so far in the richer-data `1.5B` branch.

Goal of this snapshot:

- keep a frozen copy of the code before new experiments change it
- keep the exact training config and dataset that produced the current best run
- keep the full checkpoint series for later comparison and rollback

What is included:

- `src/`
- `scripts/`
- `configs/`
- `data/`
- `tests/`
- `research_logs/`
- `README.md`
- `pyproject.toml`
- selected checkpoint files under `checkpoints/`

Main branch preserved here:

- continual run:
  - `checkpoints/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1/`
- recovery dependency:
  - `checkpoints/recovery_oasst1_1layer_1p5b_mini/memory.pt`
- background stats dependency:
  - `checkpoints/background_oasst1_1layer_1p5b_mini_stats.pt`

Primary config for the preserved best-so-far run:

- `configs/continual_oasst1_1layer_cpu_qa_fullmem_lr003_1p5b_richdata_v1.yaml`

Primary dataset for that run:

- `data/continual_identity_rich_interactions_v1.jsonl`

Notes:

- The full checkpoint series is preserved because the best checkpoint may not be the final one.
- In the current analysis, later checkpoints in this branch improved role consistency more than earlier tiny-data runs, but still had malformed outputs.
- Use the preserved checkpoints to re-evaluate or compare against future branches.

Related context:

- `research_logs/2026-05-01-identity-memory-research.md`
- `CHECKPOINTS.md`
- `MANIFEST.sha1`
