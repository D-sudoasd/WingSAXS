# Verification record for the JAC preparation changes

Date: 27 September 2026. Platform: Windows, Python 3.13.

## Existing regression suite

`py -3.13 -m pytest -q` completed with **1224 passed, 7 skipped** in 638.44 s.
The test collection preceded the addition of the new validation tests, which
are checked separately. The skipped checks require a native Qt Quick 3D
graphics backend or unmounted native/experimental fixtures. Ten warnings
were deprecation warnings from pytest-qt and Pillow; shutdown also reported
one uncollectable object. No failed test was reported.

This verifies software behavior under the local environment. It does not
verify experimental accuracy, all supported Python versions, remote CI or
independent-user success.

## New validation tools and manuscript

The final combined focused test run for trajectory comparison, uncertainty
scoring, sequence diagnostics and manifest auditing completed with
**35 passed** in 5.50 s. Tests cover
paired-input identity, scoring, interval denominators, retaining candidates
when resampling fails, malformed evidence records, nested file references and
protection against overwriting referenced inputs. The production-pipeline
sequence test also verifies that an incoming previous-frame seed can be
replaced by the current-frame algebraic initialization, rather than mistaking
parameter propagation for a different effective optimizer start.
Ruff passed across `src`, `tests` and `scripts`; `git diff --check` passed.

The final comparison ran 96 traces over 24 paired variants with no execution errors.
The additional Cartesian baseline preserved all original input hashes and
the three earlier methods' numerical records. The four-panel q-space figure
and the interval figure were visually inspected.
The uncertainty campaign completed 10 independent image trials with 32
resamples each, preserving all attempts in its coverage denominators. The
pending experimental template audit returned exit 1 and `pending_evidence`,
with scientific and independent-user outcomes `NOT_ASSESSED`, as intended.
Numerical results and limitations are in [numerical_results.md](numerical_results.md).

The final Word export is 393,887 bytes. All 68 body paragraphs matched the
Markdown source after formatting normalization. It contains 17 editable
Office Math equations, two embedded PNGs with valid relationships and alt
text, and no residual TeX commands. Full DOCX XML-schema and relationship
validation passed. Word/LibreOffice rendering was unavailable, so page layout
has not been verified by rendering. The figure source images were inspected
separately; this is not a substitute for a final Word page check.

No experimental data were supplied. No submission, public upload, release,
commit or push is represented as completed by this record.
