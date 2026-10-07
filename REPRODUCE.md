# Reproduce the revised paper figures

Run these commands from the release root. Write new figure and build outputs outside that root. The commands in this section read saved results. They do not generate observations or reevaluate a method.

## Create the figure environment

The recorded study used Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, and Matplotlib 3.9.4. The complete recorded package list is `environment/requirements-complete-study.txt`.

```sh
python3.9 -m venv ../reproduction_environment
../reproduction_environment/bin/python -m pip install -r environment/requirements-complete-study.txt
```

Exact scientific output comparison requires the recorded numerical package versions. PDF container bytes can depend on the renderer and TeX dependencies.

## Generate the revised figures

Use a new output directory. The figure helper reads `results/current/operating/complete_counts_with_design.csv`. It uses 135 saved outcome rows and analytical calculations. The saved pointwise 95% rejection intervals and full denominators remain in the displays.

```sh
../reproduction_environment/bin/python code/article_figures/make_article_figures.py --package-root . --output ../revised_paper_figures
```

The helper writes five PDF figures, SVG and PNG copies, selected counts, analytical values, and input checksum metadata. The analytical threshold is a population-variance reference. It is not a fitted threshold or a new observed power result.

Three supplement figures crop original saved PDFs. The crop removes an isolated explanatory footer. It preserves the scientific display. Create a separate environment for this helper.

```sh
python3.12 -m venv ../figure_crop_environment
../figure_crop_environment/bin/python -m pip install -r code/article_figures/requirements-crops.txt
../figure_crop_environment/bin/python code/article_figures/crop_display_footers.py --package-root . --output ../revised_paper_figures
```

The recorded crop environment used Python 3.12.14, pypdf 6.10.0, and pdfplumber 0.11.9. The helper reads `source_confidence_null.pdf`, `source_confidence_strong.pdf`, and `gain_controls.pdf` in `results/current/operating/`. It checks that the crop does not intersect another text label.

Both helpers can replace existing outputs. Use a new path. Compare the scientific CSV and JSON outputs with `paper/source/figures/`. Compare complete rendered PDF pages. Container-byte differences alone do not prove a scientific difference. See `code/article_figures/README.md` for the figure definitions and assumptions.

## Read all decisions and record metadata

The large tables use gzip compression. These commands restore the exact original bytes. They do not calculate new decisions.

```sh
gzip -dk study/analysis_001/decisions.csv.gz
gzip -dk study/analysis_001/source_sets.csv.gz
tar -xzf study/run_001_metadata.tar.gz
tar -xzf earlier_studies/record_metadata.tar.gz
```

Use an extracted working copy when you need these decompressed files. The metadata archives contain relative paths from the release root. They contain no raw trajectories. The current archive contains six metadata files for each of 20,800 records and two run-level files. The member manifest is `study/run_001_metadata_manifest.jsonl.gz`. The earlier member manifest is `earlier_studies/record_metadata_manifest.jsonl.gz`.

`COMPLETE.json` lists original raw arrays and numerical files. Some listed files are omitted from this compact release. A completion receipt is evidence of the saved execution. It is not a statement that every original file is distributed here.

## Check portable study inputs

This command checks the portable frozen inputs. It returns before random generation.

```sh
../reproduction_environment/bin/python code/study/run_study.py --registry code/study/REGISTRY_001.json --freeze code/study/FREEZE_001.json --verification code/study/INPUT_CHECK.json --output ../unused_validation_run --cutoffs code/source_inference/records/cutoffs_001 --validate-only
```

The expected result reports 69 cells and 20,800 records. A changed registry or input freeze causes refusal. The portable freeze records the original experimental freeze identity. It does not replace that original lock.

The cutoff certificates are in `code/source_inference/records/cutoffs_001/`. Their original probability integrals used Python 3.12.14, NumPy 2.5.3, and python-flint 0.9.0. The study reads the saved cutoffs. It does not recompute the integrals.

## Build the paper

The supplied paper used Tectonic 0.17.0. Copy the source to a new directory before compilation.

```sh
mkdir ../paper_build
cp -R paper/source/. ../paper_build/
cd ../paper_build
SOURCE_DATE_EPOCH=1790985600 tectonic --keep-logs manuscript.tex
SOURCE_DATE_EPOCH=1790985600 tectonic --keep-logs supplementary_information.tex
```

Tectonic needs the dependencies specified by the source. The commands produce `manuscript.pdf` and `supplementary_information.pdf` in the new directory. Exact PDF bytes can differ across platforms and dependency versions.

## Understand other workflows

The release supplies all 59 original saved display outputs and their complete companion tables in `results/current/`. It also supplies the original display and analysis code. The original `reproduce.py` diagnostics stage needs `diagnostics/diagnostics.jsonl.gz`, which this compact release omits. The three-stage original workflow therefore cannot complete from this release alone. Earlier saved-certificate extraction also needs the omitted per-record certificates. Use the supplied saved summaries for those results.

The current generation and analysis code is complete. The command below starts a new synthetic study. Do not use it to reproduce a saved figure. It needs substantial compute time and storage. The runner plans 234,946,950,400 bytes for primary uncompressed arrays, plus a separate 16 GiB auxiliary allowance. It permits one or two workers and checks a 30 GiB reserve.

```sh
../reproduction_environment/bin/python code/study/run_study.py --registry code/study/REGISTRY_001.json --freeze code/study/FREEZE_001.json --verification code/study/INPUT_CHECK.json --output ../new_synthetic_run --cutoffs code/source_inference/records/cutoffs_001 --workers 2
../reproduction_environment/bin/python code/study/analyze_study.py --registry code/study/REGISTRY_001.json --run ../new_synthetic_run --output ../new_synthetic_analysis
```

Use new output directories. Record the actual software versions and platform. The same seed can produce different floating-point arrays on another platform. Exact saved-array reproduction and reproduction of the statistical method are different checks.

Earlier studies used separate settings. Do not substitute the current registry. Their code, settings, support tables, and saved summaries remain in `earlier_studies/`. The supplement identifies reuse of earlier saved observations and summaries made after data access.
