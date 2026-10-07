# Time-reversal inference with uncertain observation channels

This compact release contains the code and saved results that support the paper by Dev D. Goyal. Read the manuscript in `paper/manuscript.pdf`. Read the proofs, assumptions, study methods, and additional results in `paper/supplementary_information.pdf`.

The release contains data needed for the figures and tables. It includes all current decision rows, complete summary tables, and metadata for all 20,800 current study records. Metadata include the seeds, generator states, settings, completion times, and original file identities. Compressed tables and metadata retain their exact original bytes after decompression. Follow [REPRODUCE.md](REPRODUCE.md).

## Release access

Source repository: [time-reversal-inference](https://github.com/dvgyl/time-reversal-inference).

Version 1.0.0 archive: [10.5281/zenodo.23223118](https://doi.org/10.5281/zenodo.23223118).

## Find the files

| Path | Contents |
| --- | --- |
| `paper/` | Manuscript, supplement, source, figures, and paper tables. |
| `code/` | Scientific methods, study generation, analysis, display code, and revised paper figure helpers. |
| `environment/` | Recorded numerical package versions. |
| `study/REGISTRY_001.json` | Current study conditions, repetitions, and settings. |
| `study/analysis_001/` | All current decisions, source sets, summaries, paired comparisons, and analysis metadata. |
| `study/run_001_metadata.tar.gz` | Metadata from all 20,800 current study records and the run start and completion records. |
| `results/current/` | All saved current operating displays, diagnostic displays, complete companion tables, and compact publication tables. |
| `certificates/source/` | Small source-power and recording-cost calculations. |
| `code/source_inference/records/cutoffs_001/` | Certified probability cutoffs used by the study. |
| `earlier_studies/` | Earlier scientific code, settings, analysis tables, figure support, saved summaries, and compact record metadata. |
| `diagnostics/` | Diagnostic scope, provenance, first-failure counts, and source-coverage counts. |
| `provenance/` | Full local manifest, omitted-file list, and an explanation of their scope. |
| `PUBLIC_FILE_MANIFEST.jsonl` | SHA-256 identities of the actual public files. |

The earlier results include the complete saved figure and table summaries used in Supplement Section S12. The record archive retains available generation and execution metadata. The saved underflow input and its provenance retain the recorded numerical failure.

## Understand the scope

The full saved local package has about 249 GB of files. This public release omits raw trajectory arrays, calibration arrays, per-record numerical certificates, and the 35.9 GB compressed diagnostic rows. `provenance/OMITTED_LOCAL_FILES.jsonl.gz` identifies these omitted files. The original package manifest also lists them. A listed checksum does not mean that the file is in this public release.

The code can generate new synthetic records from the saved registry. The original trajectories remain local. A new execution does not guarantee identical floating-point arrays on another platform. This release supplies the recorded tables and decisions directly.

The revised paper figures use analytic calculations and a small saved summary CSV. They can be reproduced from this release. The original three-stage workflow for all 59 saved displays needs the omitted diagnostic rows. Earlier saved-certificate extraction also needs omitted certificates. The release supplies those saved display outputs and summaries directly.

A decision can be rejection, nonrejection, or abstention. Nonrejection and abstention do not establish source reversibility. The unconditional rejection fraction includes every declared record. The rejection fraction among available tests is descriptive.

The results retain assumption violations, missing values, unavailable certificates, and adverse findings. The supplement identifies prospective studies and summaries made after data access. Exact arithmetic applies to the recorded inputs. It does not establish a continuous Gaussian source law. Recording-error bounds do not establish that source law. The inference guarantees require the stated source, response, covariance, calibration, and recording-error assumptions.

The registry has an inaccurate bootstrap block-length string. The executed code uses the resampled sequence length. The lag-product length is `N - 2`. The imaginary-coherency length is the number of complete segments. The code averages every value in the sampled complete blocks. It does not truncate the joined blocks to the sequence length.

`CODE_PATH_MAP.json` and the other scientific path maps identify portable code changes. The portable input freeze differs from the original experimental lock. The earlier base code retains its generator functions and omits the original locked runner. Its source-class cost routine uses a saved-input guard in place of the original multi-lock gate. These changes do not change the numerical calculations.

## Reuse and citation

The original code uses the MIT License. The original data and documents use CC BY 4.0. Third-party files retain their original terms. See [license scope](LICENSE.md) and [citation metadata](CITATION.cff).
