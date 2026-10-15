# Checkpoint snapshot construction: scalar-list copies and reliable timing

## Scope and baseline

Baseline: `cc0de47e0c1ba51831ae61bd710e115952e08384` (PR #9).
The real 121-frame records were not accessible in this validation environment.
All new timings below are **synthetic replay measurements**, not measurements
or extrapolations of the 121-frame scientific production run. No completed
scientific result was fitted again.

The production change adds two exact-built-in fast paths to `_checkpoint_safe`:
plain lists copy scalar leaves directly, and plain dictionaries avoid irrelevant
array introspection. Nested containers are still copied for every snapshot;
unknown objects, subclasses, arrays, dataclasses and non-finite values retain
the original conversion paths. A mutable historical result is never cached
by object identity. Callback changes to prior results, including changes from
integer to boolean or positive to negative zero, remain visible on the next
checkpoint.

All history, v1 fields, precision, commit frequency, flush/fsync, close and
atomic replacement are retained. Input/configuration/native-array checks are
unchanged. The previous compact writer's LF/CRLF source pairs are explicitly
added to the reviewed compatibility list alongside the existing pretty-JSON
predecessors. Current batch and builder bytes remain source-pinned. Other
calculation sources, dependency versions and analyzer identity must match.
New receipts retain their actual source identities; retained native arrays
keep their original producer provenance. Unknown later source edits and
reverse migration are not accepted by this forward-only compatibility path.

## A measurement defect found before optimization

PR #9's replay instrument replaced the globally recursive `_checkpoint_safe`
with a timer. Every scalar recursively entered the timer again. This both added
clock/counter work and summed overlapping recursion durations. It did not
change scientific values, but its stage totals were not disjoint costs.

An isolated synthetic conversion probe (12 profiles, two 4096-value lists per
profile; three trials) took median 0.01167 s without wrappers and 0.10524 s
with the old recursive timer. A wrapper confined to the outer call took
0.01249 s. This probe demonstrates measurement interference; its ratio must
not be applied to the earlier real-data measurements. In particular the
previous 205–236 s cannot be assigned entirely to production snapshot work.

The existing benchmark now binds recursion to a private globals copy of the
unchanged conversion function and times only the outer call. File write,
explicit flush, close, fsync and replacement are measured separately. The gap
between `_checkpoint_payload` returning and `write_checkpoint` starting
measures the loop that reconstructs retained future records during partial
resume (plus small call-boundary overhead). Those gap times are not added to
the inclusive payload time. `--timing outer` disables conversion/I/O wrappers
while retaining only payload/write boundary measurements, to check timing
interference. Both source versions are measured with this same updated driver.

## Comparable replay results

Synthetic fixture: 24 distinct frame records, each with eight profiles and two
2048-value scalar lists per profile; signed zero, integers, booleans and nulls
are also present. Each callback returns its own stored frame's result. The
fixture and input paths are identical between source versions. Fresh processes
run serially in alternating baseline/candidate order; numerical threads are
fixed to one. Tables give the median of three runs per source and scenario.
Individual component medians need not sum to the median total.

Environment: Linux x86-64 container, Python 3.13, same installed dependencies
and filesystem for both arms. The container's very small fsync wait is not a
measurement of production storage latency or Windows physical disk writes.

### Continuous replay: 25 synchronous commits

| Stage | Baseline (s) | Optimized (s) |
| --- | ---: | ---: |
| History result conversion, inside payload construction | 1.31769 | 0.40473 |
| Other payload construction, including metadata/config handling | 0.00561 | 0.00526 |
| **Inclusive payload construction** | **1.32330** | **0.40951** |
| Writer JSON-safe conversion | 1.29385 | 1.27442 |
| JSON encoding | 3.04264 | 3.07753 |
| File write + flush + close + fsync + replacement | 0.16553 | 0.17901 |
| **Entire instrumented `run_batch` replay** | **6.03013** | **5.09006** |

Payload construction decreased **69.05%**; total replay time decreased
**15.59%**. The writer itself was not changed. Approximately 159.05 MB of
cumulative serialized snapshots was written by both versions; tiny byte-count
differences are from independently measured elapsed-time strings.

With conversion/I/O wrappers disabled, total `run_batch` time was
**6.19384 -> 5.17596 s (16.43% lower)**. Including process startup, fixture load,
readback validation and report writing, the corresponding outer process medians
were 7.70561 -> 6.69417 s. Neither measure includes fitting, report generation
or scientific package production.

### Partial resume: 18 retained frames, six retries, seven commits

The seed checkpoint is created before timing. Every fourth record starting
with the first is marked missing; normal input/configuration/producer checks
are still executed on resume. The six callbacks replay already stored results;
they do not fit data. Both versions use the existing seven-commit semantics.

| Stage | Baseline (s) | Optimized (s) |
| --- | ---: | ---: |
| Inclusive payload construction | 0.38063 | 0.11165 |
| Retained future-record reconstruction between payload and write | 0.27978 | 0.08681 |
| Writer JSON-safe conversion | 0.64989 | 0.61841 |
| JSON encoding | 1.47892 | 1.45226 |
| File write + flush + close + fsync + replacement | 0.09752 | 0.10829 |
| **Entire `run_batch` resume replay** | **3.15929** | **2.53265** |

Total partial-resume replay time decreased **19.84%** in this synthetic case.
The native-artifact recovery contract is tested separately; the synthetic
replay does not claim to simulate production NPZ size or disk latency.

## Equivalence and recovery evidence

All nine paired replay comparisons passed strict nested field/type/`float.hex`
checks, including resumed flags. Real elapsed-time fields were excluded only
from these timed comparisons; each producer was checked against its own frozen
source hashes.

An additional independent four-frame replay loaded each source tree in its own
process, froze the batch clock and generated actual `StreamingBatchExporter`
outputs. All five complete checkpoint generations matched field/type/float
bits, including elapsed-time fields, apart from the separately verified actual
producer identities. Eight native arrays matched dtype, shape and raw bytes;
NPZ metadata was identical. Frame details, frame summary, parameter/peak/lobe
CSV files, ellipse JSON/JSONL and the evolution PNG matched byte for byte.
Manifest/provenance differences are limited to their actual source identities
and creation timestamps; commit markers retain their own generation IDs and
validated artifact digests. No scientific field was removed to make the
comparison pass.

Regression coverage includes normal continuous execution; both pretty and
previous compact checkpoint recovery; complete and partial native recovery
without refitting delivered frames; cancellation and retained later results;
input/configuration/source/dependency/analyzer/native-evidence rejection;
fsync/replace failures; actual child-process exits just after fsync and atomic
replacement; unchanged-bundle reuse; mutable historical results; scalar types,
float bits, container subclasses and detector-array omission. Existing PR #9
fixture, replay, storage and native-recovery tests are reused and extended.

## Reproduce

Use one dependency environment for both source trees. Preserve source bytes;
do not mix a baseline `batch.py` with a different baseline `producer.py`.
Run the updated driver from the optimized working tree for **both** arms:

```bash
git worktree add --detach ../WingSAXS-baseline cc0de47e0c1ba51831ae61bd710e115952e08384
python scripts/benchmark_checkpoint_writes.py sample --output-dir /tmp/checkpoint-sample

python scripts/benchmark_checkpoint_writes.py run \
  --fixture /tmp/checkpoint-sample/fixture.jsonl --source ../WingSAXS-baseline \
  --output-dir /tmp/checkpoint-before --mode batch-replay --timing stages
python scripts/benchmark_checkpoint_writes.py run \
  --fixture /tmp/checkpoint-sample/fixture.jsonl --source . \
  --output-dir /tmp/checkpoint-after --mode batch-replay --timing stages
```

Use new output directories for each repeat. Repeat with `--timing outer`;
for partial resume add `--resume-every 4` with `--mode batch-replay`.
Per-commit measurements include independent write, flush, close, fsync and
replace timings; inclusive totals and readback evidence are in `report.json`.
The existing `prepare`, `prefix` and `single-snapshot` modes remain available
for accessible real records without refitting.

## Remaining limits

Complete histories are still copied, encoded and synchronously rewritten at
each original commit boundary. Cumulative work and serialized volume therefore
still grow quadratically with frame count when per-frame record size is fixed.
JSON-safe conversion in the writer and JSON encoding now dominate this sample.
A reduction in these costs would require a separate, verified change; this
patch makes no claim to eliminate full-snapshot complexity or to deliver the
same percentage improvement on a real 121-frame scientific batch.
