# Revised paper figures

This script draws five figures for the revised paper. It does not generate observations or evaluate a statistical test. It reads the saved outcome summary and retains all three outcomes and the full denominators. The error bars use the saved pointwise 95% rejection intervals.

Run this command from the scientific package root. Use an output directory outside the scientific package:

```sh
python code/article_figures/make_article_figures.py --package-root . --output ../revised_paper_figures
```

The script requires NumPy and Matplotlib. The verified local run used NumPy 2.0.2 and Matplotlib 3.9.4 with Python 3.9.6. It uses no SciPy functions. The saved study environment can reproduce the figures.

The input is `results/current/operating/complete_counts_with_design.csv`. The script records its SHA-256 checksum in `article_figure_metadata.json`. It also writes `operating_compact_counts.csv`, which contains the 135 plotted method and condition rows. The original operating figures remain available.

The outputs are:

- `equilibrium_sensor_example`: a sensor schematic, exact equilibrium cross-covariance, and its part that changes sign under time reversal.
- `mechanism_compact`: the calibrated detector allowance and the three-channel phase cancellation.
- `source_operating_compact`: four methods at all nine strong-delay lengths and persistence 0.995.
- `cycle_operating_compact`: three methods at all nine lengths in both equilibrium and rotating conditions.
- `brownian_operating_compact`: five methods at all three lengths and all three temperature ratios.

Each figure has PDF, SVG, and PNG outputs. The analytical equilibrium calculation also writes `equilibrium_sensor_example.csv`. Use PDF files in the paper.

The equilibrium example uses drift matrix `[[1, 0.5], [0.5, 1]]`, diffusion matrix `[[1, 0], [0, 1]]`, and sampling interval one. The matched responses are both one. The small-echo responses are `H1=1` and `H2=1+0.03 exp(-i omega)`. The covariance convention is `Cij(k)=Cov(Yi(t+k),Yj(t))`. Both displays use the same source variance scale. For the echo, `C12(k)=C12_source(k)+0.03 C12_source(k+1)`. The stationary covariance and matrix exponential give these values directly. No sampled path enters this calculation.

The source horizontal axis is `(N-1)(1-phi)` with `phi=0.995`. It uses the approximate correlation time `1/(1-phi)` samples. The cycle axis is `N log(4)`, using the decay time `1/log(4)` samples. The Brownian groups use `N/2`, because the slowest drift relaxation time is two samples. Brownian method codes are C for calibrated, P for phase ignored, I for independent windows, B for moving-block bootstrap, and H for imaginary coherency.

AI assisted with the figure code. The covariance values were checked against the stated analytical model. The saved outcome counts, denominators, and intervals were retained. The plotted figures were inspected. No new observed power estimate is included.

The script also writes `analytical_model_references.csv` and `analytical_threshold_reference.json`. These files give the exact-model phase supremum, normalized contrasts, entropy rates, sufficient per-model covariance bounds, and a threshold on the population variance scale. They do not give a fitted threshold or a new observed power estimate. The per-model bound follows from `||C_X(k)|| <= lambda_max(Sigma) exp(-|k|/2)`. Summation over all lags and the detector norm give `q <= lambda_max(Sigma) (1+exp(-1/2))/(1-exp(-1/2)) max(1/v1,1.03^2/v2)`. The original study uses the shared family bound 56.

## Supplement figure footers

Three original operating PDFs have an isolated exterior note about coincident curves. The revised paper uses cropped display copies. This command reproduces those copies from the original saved PDFs:

```sh
python code/article_figures/crop_display_footers.py --package-root . --output ../revised_paper_figures
```

The crop helper requires `pypdf` and `pdfplumber`. The checked crop environment used Python 3.12.14, pypdf 6.10.0, and pdfplumber 0.11.9. Install the two direct dependencies from `code/article_figures/requirements-crops.txt`. It finds the exact footer and refuses a crop that intersects another text label. It changes the page boundary only. The original data, curves, axes, and legends remain. The original PDFs remain in `results/current/operating/`. The helper writes a crop record with the original file checksums and crop coordinates. Before and after displays were inspected. The other five supplement figure PDFs contain no such footer and need no crop.

Use one new output directory for both helpers. The helpers can replace files in an existing output directory. Preserve previous outputs before another attempt. The scientific CSV and JSON outputs supplied with the paper are in `paper/source/figures/`. Compare them with the new outputs.
