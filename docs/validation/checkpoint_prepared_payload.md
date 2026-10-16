# Prepared checkpoint payload integration after PR #11

Base: a911c2e9946652fe742e4dd054ebf61080ff3ee5. Supersedes PR #10; do not merge its old branch.

## Code comparison and retained scope

| Area | Decision |
| --- | --- |
| Scalar list/dict conversion | Keep #11 exact-built-in specialization and subclass fallback. Do not reintroduce #10 alternate dispatch. |
| Internal payload / JSON | Retain #10 prepared payload marker, normalize frame metadata and run metadata at construction, skip the writer's duplicate generic walk only for that internal payload. Public mappings still use generic conversion. |
| Array summary | Normalize overflowed min/max to null at construction, preserving the previous two-stage stored result. |
| Partial recovery | Normalize retained-record metadata before inserting it into a prepared payload. Rebuild history each time; no identity cache. |
| Recovery compatibility | Keep #11 builder checks and previous pretty/compact pairs, add exact LF/CRLF source pairs from #11 and #10, pin the resulting batch and builder. All other identity checks remain strict. |
| Performance instrumentation | Keep #11 benchmark unchanged, including nonrecursive timers and outer-only cross-check. Do not import #10 alternative benchmark or overwrite #11 evidence. |
| Tests | Keep #11 tests; add prepared/public equivalence, mutable history, scalar/array boundaries and failure tests from #10. Exercise every accepted predecessor pair in full/partial native recovery, plus actual process exits for public and prepared writes. |

The v1 scientific fields, float precision, checkpoint frequency and synchronous write/flush/fsync/close/replace are unchanged. Native NPZ bytes and producer provenance are retained by existing recovery tests. No fitting or scientific model code changed.

## Validation on 2026-10-16

Python 3.12.14, Linux. Focused run: **170 passed, 3 skipped** (Win32 metadata reader):

```bash
PYTHONPATH=src python -m pytest -q tests/test_checkpoint_prepared.py tests/test_checkpoint_snapshot.py tests/test_checkpoint_storage.py tests/test_checkpoint_benchmark_tools.py tests/test_checkpoint_quality.py tests/test_batch_export.py tests/test_p2_io_batch_contracts.py tests/test_batch_delivery_performance.py tests/test_unattended_batch.py
python -m ruff check --isolated --select E4,E7,E9,F src tests scripts
git diff --check
```

Explicit Ruff rule selection avoids host-wide style settings; these are Ruff's default rules used by the repository CI. Both checks pass. A first test run lacked pyFAI; the dependency was installed and the complete focused command then passed.

Independent baseline/candidate subprocesses with fixed clocks produced four checkpoint generations strictly equal in every nested field/type/float.hex after removing only their actual producer identity. Timed replays independently verify scientific records and each actual producer identity.

## Synthetic replay observations

Same 24 distinct records, eight profiles per frame, two 2048-point lists per profile; the unchanged #11 sample/driver is used for both sources. Serial fresh processes, same interpreter and filesystem. No fitting, report or package generation. Single paired observations, not a repeated median or real-production estimate.

| Scenario | #11 main (s) | Prepared payload (s) | Commits each |
| --- | ---: | ---: | ---: |
| Continuous run_batch, outer timers | 15.10987 | 8.96429 | 25 |
| Partial resume, stage timers | 6.25683 | 6.59625 | 7 |
| Partial writer generic conversion only | 1.70838 | 0.00783 | 7 |

Continuous replay is 40.67% shorter in this observation. Partial stage-timed total did not improve despite removing generic conversion: stage instrumentation also observes newly added metadata-boundary calls, and single-run timing is variable. Do not claim a general partial-resume speedup from it. All four runs passed strict readback; partial runs reused 18 frames and retried six. Accumulated complete snapshots remain O(N²).

```bash
python scripts/benchmark_checkpoint_writes.py sample --output-dir /tmp/checkpoint-fixture
python scripts/benchmark_checkpoint_writes.py run --fixture /tmp/checkpoint-fixture/fixture.jsonl --source BASELINE_TREE --output-dir /tmp/before --mode batch-replay --timing outer
python scripts/benchmark_checkpoint_writes.py run --fixture /tmp/checkpoint-fixture/fixture.jsonl --source CANDIDATE_TREE --output-dir /tmp/after --mode batch-replay --timing outer
```

For partial recovery add `--resume-every 4`; use `--timing stages` for disjoint conversion/I/O counters. Real 121-frame production records were unavailable. These results must not be combined with #10/#11 percentages.

## Coordination and publication

#10 head remained 1d17971b993e8f42cc5f908917d183d86a29e1a0 with no discussion updates; #11 closeout explicitly left it untouched. No public evidence of ongoing branch work was found. The connected workspace API was unavailable; no claim is made about unseen remote Agent sessions. All work used an isolated clone; neither old branch nor another working environment was modified. Main remains unchanged pending review/CI of this replacement.
