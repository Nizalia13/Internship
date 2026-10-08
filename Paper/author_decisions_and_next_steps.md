# Decisions and checks for manuscript v0.1

Prepared 28 September 2026. This is an editing companion, not part of the submission.

## Working direction

Provisional framing: a methods-and-software paper on class-specific aggregation of incomplete predictor scores. The biological study motivates the method; synthetic, Iris, and football examples demonstrate portability. The draft is deliberately independent of a journal template.

The report can supply problem motivation, score construction, and the biological protocol. It cannot simply be shortened verbatim: some mathematics and implementation details have changed. In particular, distinguish a strip score sublevel set from a geometric dilation; retain alpha/d as an intentional shape choice; do not label collapsed mean regions as intrinsically non-convex; and do not present class-conditional calibration as a newly invented general principle.

## Version and evidence ledger

- Linked repository HEAD was checked directly: `e60c4ecbbbc8393be80a23d2468af0eee99c3f42`, message “Updated football position classification notebook”. A separate snapshot is in `tmp/paper-library-source`.
- Local Internship repository HEAD was `95b56f9`, message “updated the examples”. It is a different repository and is not a substitute for the remote library version.
- Envelope modules, calibration, missingness, and metrics match between the local library and downloaded remote snapshot. Model and plotting files differ.
- Current remote synthetic single run: seed 13, alpha 0.10. Do not reuse the earlier seed-21 single-run results as if they were this run.
- Current remote synthetic repeated runs: seeds 7, 21, 25, 73, 99. Saved means were transcribed; full clean execution still required.
- Current remote Iris: seed 25, alpha 0.10; saved coverage 24/30, 25/30, and 23/30. The local Iris notebook has alpha 0.30, so its values must not be substituted.
- Football: four broad classes and four base classifiers. Saved outputs were inspected, not independently rerun for this draft.
- The thesis host results refer to an earlier implementation and test split 0. Prior local coverage diagnostics refer to an earlier library and validation split 1. Neither is a clean current-release benchmark.

## Decisions for the authors

1. Choose the primary contribution: new score constructions, reproducible scientific software, or a biological application. Decide what constitutes the central comparative claim.
2. Confirm author list, ordering, affiliations, corresponding author, funding, and contributions. The thesis names a supervisor, but that does not settle paper authorship.
3. Choose a journal after matching the contribution and evidence to its scope. No journal has yet been selected or contacted.
4. Decide whether the biological case belongs in the main paper, supplementary material, or a separate application paper.
5. Confirm whether there is a public thesis record to cite and whether the biological scores/data can be released.

## Necessary implementation and theory alignment

- Freeze score construction before final S2 calibration. Resolve pre-split column selection for collapsed and radial.
- Specify fully missing candidate behavior as a population-level policy used consistently in calibration and prediction; avoid silently conditioning training on score availability while claiming coverage for a broader deployment population.
- Document that alpha/d is a shape heuristic; the final rank step uses alpha. Do not claim a Bonferroni proof over all strip constraints without deriving it.
- Keep the raw strip score and threshold comparison as the formal method. If a literal geometric gauge is intended, that requires a different derivation/implementation and a fresh evaluation.
- Audit zero-limit regularization, numeric stability, and the radial shape/query subspace conventions. Fix any chosen regularization from S1, not final test performance.
- Retain the correct infinite-threshold rule for small calibration sets and report how many classes it affects.
- Clarify model selection: tune upstream of a fresh final calibration stage rather than assume that any use of a validation set automatically protects S2 independence.

## Experiments needed for a submission claim

1. Re-execute the notebooks in a pinned environment and export prediction sets, split IDs, score tables, parameters, and source hashes. Preserve the current saved outputs as historical evidence until reproduced.
2. Expand synthetic evaluation beyond five seeds. Predeclare repeated independent training/calibration/test draws and vary predictor dependence, informative/noise dimensions, class imbalance, missingness rate, and missingness mechanism. Report between-run variability and per-class coverage. Do not select seeds for favorable coverage.
3. Compare against observed-coordinate mean/max baselines, individual predictors, and a relevant score-aggregation baseline such as CSA. Where probability-based methods are applicable, define the ensemble probability vector and evaluate an appropriate standard classification conformal baseline. Align calibration budgets and label-conditioning targets, or explicitly distinguish them.
4. Ablate alpha versus alpha/d shape quantiles, global versus local monotonicity, one-coordinate policies, radial subspace conventions, and zero-limit regularization. Keep final alpha fixed.
5. Reproduce biological data preparation or obtain sufficient provenance to establish which frozen base model generated each score. Examine score and mask distributions by fold and host; do not explain a 75% coverage result solely by nominal finite-sample variability.
6. Record runtime and memory with a fixed environment and dimensions. A single sorting step does not make the complete procedure O(n log n); score construction and prediction have their own costs.

## Proposed figure plan

1. Workflow: measurements → frozen base predictors → candidate score vectors → S1 shape construction → S2 threshold calibration → label sets.
2. Geometry: paired two-score refits for the three methods, with explicit missing-coordinate examples. Label these as diagnostic refits.
3. Main comparison: coverage and mean set size across repeated experiments, with between-run uncertainty and per-class panels.
4. Missingness/dimension ablation: performance as the number of available predictors and correlation change.
5. Optional application panel: football class-inclusion heatmap and a clearly labeled fixed-coordinate slice; historical host figures only after provenance and implementation alignment.

No figure is claimed to have been produced for this manuscript yet.

## Repository release checks

The remote root currently lacks the local pyproject.toml and an observed license file. Confirm installation metadata and license with the authors. The remote README describes an all-positions football notebook that was not present in the inspected tree. Resolve this before presenting the repository as a complete reproducible release. Tag the chosen version and archive it with a stable identifier if appropriate.

## Time estimate

- A substantive discussion draft: available now as manuscript_v0_1.md.
- A coherent author-reviewed draft: approximately 2–3 focused revision sessions, assuming the framing is settled and feedback is available.
- A submission-ready paper: provisionally 1–3 weeks if the required data are accessible and experiments run smoothly. New theoretical changes, missing data provenance, or expanded benchmarks can extend this substantially. This estimate is for collaborative work, not a promise of unattended future execution.

The fastest responsible route is to revise the paper's argument while reproducing and strengthening the evidence. Do not wait for polished figures to agree on the contribution, but do not submit numerical or coverage claims before their checks are complete.
