# Working on WingSAXS

WingSAXS (`butterfly-saxs`, import `butterfly_saxs`, CLI `bsaxs`) measures
butterfly-pattern SAXS trajectories and their evolution from observed intensity.
Python 3.11–3.13 is supported; prefer the existing `.venv-project` environment.
On Windows use its explicit interpreter when `python` resolves to the Store stub.

## Work from the current decision

- Finish implementation and appropriate verification within the user's scope.
  Ask only for missing facts that materially change scope, scientific meaning,
  acceptance or permission and cannot be inferred from available evidence.
- Start with the task, existing context and changed files. Search the relevant
  symbols and callers, then read only the needed sections. Expand when that
  evidence is insufficient; there is no mandatory document set or repo map.
- Reuse reads, parsed inputs and passed checks while their source/configuration
  is unchanged. Track this in the task context; no new agent-maintained cache.
  Batch independent reads in one tool call. Delegate only independent work that
  saves time after coordination costs.
- Select tests for the affected behavior and interface. After they pass, proceed.
  Recheck after new edits, a failure or a specific unresolved risk. Full tests,
  builds and integrity audits have explicit triggers, not a per-task ritual.
  Apply skills for the task; overlapping skills share the same validation evidence.

## Boundaries

- Fit each frame to its observed intensity. Seeds, neighbors, interpolation and
  bounds guide optimization; they never fabricate missing pixels/arcs or replace
  a failed fit with a previous frame. Retain source-frame identity.
- Keep finite estimates, candidates, failed fits and missing data distinguishable
  in UI/exports. Heuristic quality warnings retain results and identify useful
  changes to q windows, masks, calibration or fitting. Invalid shapes, empty
  domains and invalid calibration/weights need actionable errors; isolate failed
  frames and keep their sequence positions.
- Keep ring `q*` / `L = 2π/q*` separate from ellipse `a`, `b/a`, axis angle and
  conditional Ln/Lz. Physical lengths require calibrated physical q; pixel-q
  cannot produce nm. Ellipse geometry is not a unique 3D reconstruction.
  Geometry analysis does not implicitly run the separate empirical `full2d` model.
- CLI stdout is strict JSON (`allow_nan=False`, ASCII escaped); diagnostics go
  to stderr. Exit 0 = completed without quality warnings, 1 = retained warnings,
  partial failure or cancellation, 2 = input/configuration/overwrite error.
  Engineering completion and preflight/P3/P4 reports are not scientific acceptance.
- `cli.py` loads scientific modules lazily; `service.py` stays Qt-free; workers
  do not access widgets. Verify the CLI/service/UI seam actually affected.
- Keep raw inputs, PONI and masks unchanged; write outputs separately and respect
  overwrite choices. Never commit/upload `CHANGELOG.md`, `data_local/`, private
  literature or local validation artifacts. Review staged files before publishing.

## Load detail when needed

- Task routing, focused checks and deep-check triggers: `docs/agent_workflow_zh.md`.
- New environment or startup failure: `docs/first_run_zh.md` and
  `constraints/validation-py311-313.txt` (new install: `-e ".[all]"`).
- Scientific definitions/quantity interpretation: `docs/scientific_basis_zh.md`;
  butterfly tracing: `docs/butterfly_arcs_zh.md`.
- Result schema or serialization change: `docs/validation/result_schema_v1.md`;
  service/UI boundaries: `docs/architecture_zh.md`.
- Unknown CLI contract: `bsaxs describe COMMAND` or `bsaxs COMMAND --help`.
  Reuse it for unchanged installed code. Run `bsaxs doctor --json` only for a
  new/changed environment or an import failure. User procedures: `docs/user_guide_zh.md`.
