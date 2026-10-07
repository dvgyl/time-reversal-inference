# Analysis definitions

This plan is fixed before scientific simulation. The registry defines every independent record. It has 69 cells and 20,800 records. Independent records at different lengths do not share prefixes. A paired numerical or gain comparison does not add an independent record.

## Decisions and uncertainty

Each method returns reject, nonreject or abstain. The unconditional rejection fraction uses every declared record in its cell. Abstention remains in this denominator. Rejection among available tests uses only reject plus nonreject and is descriptive. An execution failure is reported separately and cannot silently become a nonrejection. A stopped run is incomplete. Its records and failures are preserved before a corrected continuation is specified.

Report two-sided 95% Clopper-Pearson intervals for rejection probabilities. For every within-cell method pair, report the two discordant rejection counts. The 95% conservative difference interval subtracts simultaneous 97.5% Clopper-Pearson intervals for these event probabilities. The sign-test probability conditions on the number of discordant pairs. It describes equality of their rejection probabilities. It does not compare power under different null hypotheses. These are descriptive pairwise intervals without a familywise claim across all reported comparisons.

Summaries include all cells. Keep model violations separate from valid nulls. For each threshold diagnostic, report its available count and the minimum, quartiles, median and maximum. A missing threshold stays missing. For a bank, the displayed member has the largest exact statistic-minus-upper-threshold margin among available members. This is a diagnostic of the declared union rule. The saved outputs contain all members.

Source diagnostics include the complete outward acceptance set, its convex hull, true-parameter coverage, finite endpoint, unresolved components, and width. Coverage is checked with rational comparisons. The persistence span is t(upper)/t(lower), where t(phi)=(1+phi)/(1-phi). Also report the minimum bank envelope and its ratio to the continuous-bank optimum when these are finite. Numerical ablations use the same acceptance-set and whole-bank decision definitions.

## Comparators

All comparators use the primary quantized observation. The lag comparators retain rows 1 through N-1, center each channel, and form U_t=Y_1(t)+Y_1(t+1), W_t=Y_2(t)-Y_2(t+1). They center U and W and average their product. With n=N-2, empirical variances v_U,v_W, and t=log(2/alpha), the independent-window radius is sqrt(v_U v_W)[sqrt(2(n-1)t)+t]/n. This is a declared naive comparator under dependent recordings.

The moving-block comparator resamples the centered product sequence. Its block length is max(2,ceil(n^(1/3))). It samples ceil(n/block_length) complete noncircular blocks with replacement. It compares the absolute resampled mean with the absolute observed mean. The probability estimate is (1+exceedances)/500 for 499 resamples. This is an approximate bootstrap test.

Phase-adjusted comparators use the same supplied or estimated phase upper bound as the calibrated procedure. Their plug-in phase tolerance is twice this bound times the geometric mean of the two empirical marginal variances. The bootstrap compares its absolute resampled mean with max(0,abs(statistic)-tolerance). These comparators abstain if a phase certificate is absent. Cycle cells have no such certificate.

Brownian and phase-boundary cells also compare the same dependence-aware threshold with the phase allowance set to zero. This deliberately ignores the relative detector phase and has no source-null size guarantee. Separate missing-response and missing-covariance procedures abstain. This distinguishes missing information from a calculation that incorrectly treats missing phase as zero.

The imaginary-coherency comparator divides observations into nonoverlapping 256-row segments. It centers each segment, computes the discrete Fourier coefficient at pi/2, and divides the average imaginary cross product by the geometric mean spectral scale. It applies the declared moving-block test to the segment contributions. An incomplete final segment is discarded for this comparator only. Fewer than four complete segments or a zero spectral scale cause abstention. It tests an observed-signal target. It is not a test of source reversibility under unknown unequal responses.

Each record has one separate bootstrap stream. Comparators consume it in a fixed order. Raw lag, phase-adjusted lag when available, then imaginary coherency use sequential parts of this stream. There is no search over block length, frequency, calibration, lag or thresholds.

## Numerical and recording comparisons

Primary source inversion uses continuous quadratic root enclosures with 48 root bits. The comparison uses dyadic subdivision depths 20, 28 and 36. Arithmetic comparisons use 80, 160 and 320 square-root bits with the same 48-bit root enclosure. The exact compiled moments do not depend on those settings and can be reused with a recorded identity change. Calibration and the entire bank are reevaluated under each precision.

The primary recorder has 20 fractional bits. Paired recording variants retain calibration at this precision. The fractional 8-bit and 12-bit grids isolate rounding. Finite 8-bit and 12-bit recorders cover [-16,16), with steps 32/2^bits and outer bin-center clipping. A value outside that range is counted as saturated. Without a finite perturbation certificate, all bound-based decisions in that recorder variant abstain.

## Execution and preservation

Use at most two local workers. A worker checks a 30 GiB disk reserve before opening a stream. The driver has at most two jobs in flight and stops new dispatch after a failure. A completed record can be reused only after its full file list and hashes pass verification. Resume opens only unstarted streams. An incomplete or failed record is never rerun by resume.

The whole-study storage check allows 234,946,950,400 bytes for uncompressed primary floating-point and integer arrays. It adds 16 GiB for auxiliary files and the 30 GiB reserve. The auxiliary allowance is a planning estimate. The per-draw check still applies if file sizes differ.

Save generated binary64 records before quantization. Save quantized integer arrays, calibration data, method outputs, stream states, and hashes. Exact comparisons control decisions. Floating-point summaries and plots are descriptive transformations of those saved decisions. Hashes bind the registry, method code, cutoff certificates, and this analysis plan before the first draw.

The local freeze is not an external preregistration. No public registration or deposit is established by these files.
