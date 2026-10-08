# Class-specific conformal envelopes for classification with incomplete predictor scores

**Working manuscript v0.1 — 28 September 2026**

**Authors and affiliations:** to be confirmed by the authors. Author order, corresponding author, funding, and contribution statements have not been assigned.

**Draft status.** This is a substantive first draft for discussion, not a submission-ready manuscript. It describes the repository at commit `e60c4ecbbbc8393be80a23d2468af0eee99c3f42`. Numerical example results below are transcribed from that revision's saved notebook outputs, not newly executed experiments. Thesis results are explicitly identified as historical. Author notes identify decisions and evidence still needed. The theoretical proposition specifies sufficient assumptions; it is not a blanket certification of every execution path in the current software.

## Abstract

Classification systems often produce several scores for each candidate label, reflecting different models, measurements, or sources of evidence. Combining these scores into an uncertainty-aware prediction is complicated when predictors are correlated, score availability varies between observations, and class frequencies are unequal. We present Conformal Envelopes, a Python framework that constructs a separate acceptance region for each class and returns the labels whose regions accept a new observation. Three score constructions are implemented: an observed-coordinate mean, a direction-dependent radial envelope, and a system of pairwise strip constraints. Shape discovery and threshold calibration are separated, allowing a common class-specific conformal rank rule to be applied to different score geometries. Missing coordinates are handled within the score construction rather than replaced by observed numerical values. We state the assumptions required for finite-sample class-conditional coverage and distinguish this guarantee from realized test-set coverage, missingness-pattern coverage, and coverage after candidate filtering. Repository examples illustrate simulated incomplete scores, classifier outputs for Iris flowers, and football-player role classification. A previous phage-host study motivates the treatment of structural missingness and highlights the need to audit score provenance across data partitions. The contribution is a reusable implementation and an explicit account of the trade-offs between aggregation, geometry, and calibration; the available examples do not establish universal superiority of one envelope family.

**Keywords:** conformal prediction; classification; score aggregation; missing predictor scores; class-conditional calibration; uncertainty quantification.

## 1. Introduction

A classifier commonly reports a single label even when several labels are plausible. This can conceal uncertainty that matters in subsequent decisions. A prediction set makes this uncertainty explicit: a singleton identifies one accepted candidate, a larger set preserves alternatives, and an empty set records that no candidate met the acceptance criterion. The usefulness of such a set depends jointly on whether it contains the correct label and how much ambiguity remains.

In applications with several predictors, uncertainty assessment must also decide how to combine their evidence. Different classifiers may be trained on the same features, producing strongly dependent scores. Other systems assemble scores from distinct measurements or biological functions. A score may be absent because a measurement was unavailable or because the corresponding feature is structurally absent. Replacing a missing score by zero is particularly problematic when zero has a substantive interpretation as strong or weak evidence.

Conformal score aggregation provides a relevant starting point. Ochoa Rivera, Patel, and Tewari [1] construct multivariate score envelopes rather than combine already-formed prediction regions. Their work motivates preserving information about the joint score vector. Here we investigate a practical class-specific formulation with explicit missing-score handling and several interchangeable envelope families. We do not claim that multivariate score aggregation or class-conditional conformal calibration is itself new.

The motivating application is phage-host classification, where function-specific predictor scores can be unavailable because a phage lacks proteins associated with a function. The original master's study [6] developed an application-specific workflow. The present framework separates the envelope machinery from the biological data preparation, making it usable with scores supplied by other classification systems.

The paper has three objectives. First, it defines a common interface for class-specific score aggregation, calibration, and prediction. Second, it specifies observed-coordinate mean, radial, and strip scores in a way that distinguishes statistical validity from geometric interpretation. Third, it documents examples and limitations needed to evaluate these methods responsibly, including small calibration sets, missing-score patterns, numerical degeneracy, and distribution differences between calibration and evaluation data.

**Author note — contribution:** establish with the supervisor which constructions are methodological contributions and which are implementation choices or adaptations of earlier work. Avoid a priority claim until the related-work comparison is complete.

## 2. Related work and scope

Conformal prediction calibrates a score against held-out observations to obtain set-valued predictions. Its standard finite-sample guarantee is an average probability statement under exchangeability, not a lower bound on the empirical coverage of every realized test set [2]. Classification methods such as adaptive prediction sets provide important comparative baselines [3].

CSA [1] is the closest methodological reference: it uses directional quantile envelopes to aggregate multiple scores. Our experiments should compare against its construction where the input and missingness assumptions can be made comparable. Scalar score pooling is another essential baseline because a more expressive envelope is useful only if its additional complexity improves the coverage–set-size trade-off.

Flexible non-convex conformal regions also occur in other formulations. For example, Thurin, Nadjahi, and Boyer [4] use optimal transport to construct multivariate conformal regions. Accordingly, non-convex geometry alone is not a sufficient novelty claim. The narrower focus here is a modular classification implementation with class-specific score columns, observed-coordinate evaluation, and inspectable acceptance rules.

This work concerns prediction sets over a finite label space. It does not train the underlying predictors, recover missing measurements, or identify every biologically possible host from single-label annotations. Its generality lies in accepting appropriately constructed predictor scores from multiple domains.

## 3. Problem formulation and data separation

Let the label space be \(\mathcal Y=\{1,\ldots,C\}\). For candidate class \(c\), an observation has a score vector

\[
 p_c(x)\in([0,1]\cup\{\mathrm{NA}\})^{d_c}.
\]

The dimension \(d_c\) and score columns may differ between candidates. In the default probability-like convention, larger scores support the candidate. Observed coordinates are transformed to nonconformity coordinates \(v_{cj}(x)=1-p_{cj}(x)\), while missing values remain missing. Define the observed set

\[
 A_c(x)=\{j:v_{cj}(x)\ne\mathrm{NA}\}.
\]

Lower nonconformity indicates stronger agreement with the candidate. Users may instead supply nonnegative nonconformity coordinates directly. The score transformation is fixed before calibration and used identically at prediction time.

There are two distinct learning stages. A base-training partition fits the predictors and any preprocessing. A separate conformal-fitting partition supplies score vectors and labels. For each class, the latter is divided into shape data \(S_{1,c}\) and calibration data \(S_{2,c}\). Shape data determine a scalar score function \(a_c\), and calibration data determine a threshold \(q_c\). Predictions are

\[
 \mathcal C(x)=\{c:a_c(v_c(x),A_c(x))\le q_c\}.
\]

Although geometry is learned in multiple dimensions, the final rank calibration acts on a scalar score. This makes the three constructions compatible with one calibration routine.

For the ordinary split-conformal argument, predictors, feature selection, shape construction, and hyperparameters must be fixed independently of the final calibration scores. If model selection uses previously calibrated prediction sets, the selected construction should receive fresh calibration data, or an appropriate alternative validity argument is required. Simply keeping final test labels hidden is not sufficient to resolve every form of calibration reuse.

## 4. Envelope score constructions

### 4.1 Observed-coordinate mean

For \(|A_c(x)|>0\), define

\[
 \bar v_c(x)=\frac{1}{|A_c(x)|}\sum_{j\in A_c(x)}v_{cj}(x).
\]

The implementation estimates a shape quantile \(b_c=\operatorname{Quantile}_{1-\alpha}\{\bar v_c(x):x\in S_{1,c}\}\) and uses

\[
 a_c^{\mathrm{mean}}(x)=\frac{\bar v_c(x)}{b_c+\varepsilon},
 \qquad \varepsilon=10^{-12}.
\]

The same positive denominator is applied to calibration and prediction observations. Consequently, for a fixed retained coordinate set, calibrating this score is equivalent to applying the corresponding order-statistic threshold directly to the observed-coordinate means. The shape normalization changes the reported scale but not the ranking. This is a baseline against which the more elaborate constructions should be judged.

The mean is simple and inexpensive, but discards distinctions between vectors with the same average. Correlated predictors may repeat the same evidence, and noisy coordinates can dilute informative coordinates. When missingness varies, different observations can be averaged over different subsets; this is part of the score definition and must be reflected in the population to which the validity claim applies.

### 4.2 Radial score

The radial construction uses both the magnitude and direction of the observed nonconformity vector. For a nonzero vector, write

\[
 r(x)=\|v_{c,A_c(x)}(x)\|_2,
 \qquad d(x)=v_{c,A_c(x)}(x)/r(x).
\]

Reference directions \(u_1,\ldots,u_M\) are sampled from the positive unit sphere by normalizing absolute standard Gaussian vectors. Shape observations are grouped by directional similarity. For each reference direction, an empirical radius quantile is estimated from observations in an angular neighborhood. Sparse neighborhoods are blended with a marginal radius quantile; neighborhoods without observations use that marginal quantile entirely.

More precisely, the current implementation uses a minimum support \(n_{\min}=\max(10,\lfloor\rho |S_{1,c}|\rfloor)\). If a neighborhood contains \(n_m\) observations, its local quantile receives weight \(\min(1,n_m/n_{\min})\), with the remaining weight assigned to the marginal quantile. The shape level is \(1-\alpha\).

At score evaluation, each reference direction is restricted to the query's observed coordinates and renormalized. Denote the resulting directions by \(\tilde u_{m,A}\). Smooth weights and a radius are

\[
 w_m(d,A)=\frac{\exp\{\kappa d^\top\tilde u_{m,A}\}}
 {\sum_{\ell=1}^{M}\exp\{\kappa d^\top\tilde u_{\ell,A}\}},
 \qquad R_c(d,A)=\sum_{m=1}^{M}w_m(d,A)b_{cm}.
\]

The score is \(a_c^{\mathrm{rad}}(x)=r(x)/(R_c(d(x),A_c(x))+\varepsilon)\); the implementation assigns zero to magnitudes below its numerical tolerance. For a fixed missingness pattern, the directional boundary gives a radial acceptance construction. It does not imply comparable efficiency across patterns with very different observed dimensions.

**Implementation qualification.** Shape discovery currently embeds observed directions in the full coordinate space with zeros in missing positions before comparing them to reference directions. Query evaluation instead renormalizes the reference directions in the observed subspace. These operations are not identical. Because the final evaluation score is used consistently on S2 and future queries, this distinction need not invalidate rank calibration, but it can affect the learned geometry and should be included in a missingness ablation.

### 4.3 Pairwise strip score

The strip method models a constraint on coordinate \(i\) conditional on the bin occupied by another coordinate \(j\). Columns completely absent from S1 are removed, and the same retained columns are used thereafter. Let \(d_c\) now denote this retained dimension. Shape quantiles use the intentional level

\[
 \alpha_{\mathrm{shape}}=\alpha/d_c.
\]

Each conditioning coordinate is divided into equal-width bins over its observed S1 range. For bin \(b\), the conditional bound \(L_{ji b}\) is estimated from rows where both coordinates are observed and coordinate \(j\) lies in that bin. If support is below a prescribed minimum, the conditional quantile is blended with the marginal quantile of the target coordinate; with zero support the marginal quantile is used. This construction does not impute a missing target value.

A reverse cumulative maximum imposes

\[
 L^{\mathrm{mono}}_{ji b}=\max_{b'\ge b} L_{ji b'}.
\]

Thus, moving to a smaller conditioning coordinate cannot tighten that pairwise bound. For observations with at least two observed coordinates,

\[
 a_c^{\mathrm{strip}}(x)=
 \max_{\substack{i,j\in A_c(x)\\i\ne j}}
 \frac{v_{ci}(x)}{L^{\mathrm{mono}}_{ji,b_j(v_{cj}(x))}+\varepsilon}.
\]

Values outside the learned bin range use the nearest boundary bin. With exactly one observed coordinate, the score uses that coordinate's marginal shape quantile as its denominator. This avoids the automatically zero score that would result from an empty collection of pairwise constraints.

The \(\alpha/d_c\) adjustment controls initial shape conservatism. It is not, by itself, a proof of joint coverage over all bins and ordered coordinate pairs. The final scalar score is calibrated at the original level \(\alpha\).

The strip score is generally not a geometric gauge \(\inf\{t:x\in tE\}\), because scaling a point can change its conditioning bin. We therefore describe the acceptance family as score sublevel sets rather than claim that every threshold simply dilates a fixed region. This distinction corrects an interpretation in the earlier thesis without requiring that the score cease to be useful for conformal calibration.

### 4.4 Missing observations and numerical conventions

Partially observed vectors are evaluated using the available coordinates. Neither observed-coordinate averaging nor pairwise constraint omission reconstructs the missing values. If the mask distribution changes between calibration and deployment, the distribution of the resulting score may also change.

Entirely missing candidates require separate treatment. At the reviewed revision, fitting rejects rows with no usable true-class score; prediction rejects entirely missing candidates under collapsed scoring but accepts them under radial and strip scoring. These are different abstention policies. The general coverage proposition below applies only where its population and score-definition assumptions hold; it does not automatically cover a pipeline that excludes a type of observation from calibration and later encounters it at deployment.

The fixed stabilizer prevents division by zero but can produce very large ratios when a strip limit is zero. A large common threshold may then weaken other constraints. A publication experiment should report the frequency of zero limits and assess an S1-defined regularization rule, without choosing that rule from final test results.

## 5. Class-specific calibration and validity

Let \(n_c=|S_{2,c}|\), and let \(a_{c,(1)}\le\cdots\le a_{c,(n_c)}\) be the ordered true-class calibration scores. Set

\[
 k_c=\lceil(n_c+1)(1-\alpha)\rceil,
 \qquad q_c=\begin{cases}
 a_{c,(k_c)},&k_c\le n_c,\\
 +\infty,&k_c>n_c.
 \end{cases}
\]

**Proposition 1 — class-conditional rank coverage.** Fix the score-generating predictors, coordinate selection, shape data, and any algorithmic randomness. Suppose the resulting true-class scores of the calibration observations and a future observation are exchangeable within class \(c\). Suppose also that the score is defined on the relevant population and that candidate \(c\) is not removed after the envelope test. Then

\[
 \Pr\{c\in\mathcal C(X_{\mathrm{new}})\mid Y_{\mathrm{new}}=c\}\ge1-\alpha.
\]

**Proof.** Under exchangeability, the future score has a uniform rank among the \(n_c+1\) scores when ties are randomly ordered. For \(k_c\le n_c\), the event that this rank is at most \(k_c\) implies acceptance against the \(k_c\)-th calibration score. Its probability is \(k_c/(n_c+1)\ge1-\alpha\). Using a non-strict comparison preserves or enlarges acceptance in the presence of ties. If \(k_c>n_c\), the infinite threshold accepts every score in the specified score domain. This is the standard conformal rank argument applied separately to each class, not a new general coverage theorem [2].

The probability averages over calibration and future-sample randomness under the stated conditioning. It is neither a guarantee that every realized validation set attains the nominal fraction nor a guarantee conditional on arbitrary covariates or missingness patterns.

**Implementation qualification.** Collapsed and radial currently remove all-missing columns using the full class fitting sample before the S1/S2 split. This is not the S1-only selection assumed in the proposition. Before using Proposition 1 as an implementation-level guarantee, selection should be fixed independently of S2 or justified by an additional argument. The all-missing policies likewise need a defined domain and a coherent treatment. These limitations must not be hidden by presenting the abstract theorem as a proof for every code path.

At \(\alpha=0.10\), fewer than nine calibration observations require an infinite threshold under this rule. Capping the rank at the observed maximum may produce smaller sets but loses the stated nominal guarantee. Infinite thresholds therefore represent a statistical information limitation, not simply a numerical fault.

### 5.1 Candidate filtering and empty sets

If a preliminary stage returns candidate labels \(G(x)\), the final prediction becomes \(\mathcal C(x)\cap G(x)\). If the two stages have valid miscoverage bounds \(\alpha_E\) and \(\alpha_G\) for the same target population, the union bound gives final coverage at least \(1-\alpha_E-\alpha_G\), without requiring independent errors. Using two nominal 90% stages does not automatically give a 90% final guarantee. Conditional claims require the corresponding conditional assumptions and bounds.

Adding a label only when the original prediction set is empty cannot reduce coverage because it only enlarges the set. Nevertheless, forced labels are decisions, not calibrated probabilities, and should be reported separately. The principal examples disable forcing so that empty sets remain visible.

## 6. Software implementation

The public `ConformalSetModel` interface separates loading data, transforming scores, constructing envelopes, computing thresholds, and predicting label sets. Input can be a CSV path or a pandas DataFrame with a unique ID, numerical score columns, and a training label. The `label_to_columns` mapping associates each candidate with its own predictors; absent a mapping, all classes share the supplied columns.

`fit` constructs one envelope per represented class. `predict` evaluates candidate classes and returns a DataFrame, records, or a dictionary keyed by sample ID. Outputs include prediction sets, set sizes, forcing indicators, and normalized per-label scores. These normalized scores are not posterior probabilities. `get_envelopes` exposes the stored construction, including shape/calibration sizes and indices. Evaluation reports overall and per-class coverage, average set size, singleton rate, singleton accuracy, and empty-set rate.

Two plotting operations answer different questions. `plot` refits a separate two-coordinate diagnostic model. `plot_slice` holds unplotted coordinates fixed and evaluates a cross-section of the existing fitted model. In the reviewed remote code, hidden coordinates default to zero nonconformity unless `slice_values` overrides them. Test-point colors in a slice use the full observed score vectors, so the apparent location of a point relative to the displayed boundary need not agree with its full-model category. Class-inclusion heatmaps show acceptance frequencies and must not be interpreted as ordinary confusion matrices with rows summing to one.

The repository records package version 0.2.0, but manuscript reproducibility should use the full commit identifier rather than the version string alone. Repository URL: <https://github.com/Nizalia13/Conformal_Classification_Library>.

**Author note — release:** the inspected remote tree does not contain the local `pyproject.toml`, and no license file was observed at its root. Installation instructions, redistribution terms, a tagged release, and an archival identifier need confirmation before submission. Public availability alone does not establish an open-source license.

## 7. Experimental examples and preliminary findings

### 7.1 Evaluation principles

For test observations \((x_i,y_i)_{i=1}^{N}\), report

\[
 \widehat{\mathrm{coverage}}=\frac1N\sum_i\mathbf1\{y_i\in\mathcal C(x_i)\},
 \qquad \widehat{\mathrm{size}}=\frac1N\sum_i|\mathcal C(x_i)|.
\]

Coverage alone is insufficient because returning every label covers every observation. Set size alone is insufficient because rejecting every label produces size zero. Empty sets are counted as uncovered. Singleton accuracy conditions on receiving a singleton and is not overall accuracy or coverage. Per-class results are needed to reveal performance differences obscured by class frequency.

The results below describe saved demonstrations at the reviewed revision. They establish examples to reproduce, not a completed comparative benchmark. No missing experiment is represented by a fabricated numerical result.

### 7.2 Synthetic incomplete scores

The synthetic example contains two candidate labels with four score sources each: strong, correlated, weak, and noise-only predictors. Candidate-dependent Gaussian latent variables are transformed to bounded scores, and the strong predictors share latent correlation 0.90. Training and test samples are independent generator draws, with 1,500 fitting observations and 500 test observations. Missingness is proposed independently at probability 0.15 per score, but masks with all four candidate scores missing are resampled. Thus the experiment excludes completely missing candidates and is not an evaluation of that edge case.

The current single-run cell uses seed 13 and \(\alpha=0.10\). Its saved coverage is 0.858 for collapsed, 0.880 for radial, and 0.888 for strip, with mean set sizes 0.950, 1.014, and 0.944. These are realized outcomes, not violations established solely by comparison with the nominal target.

The repeated-run section uses seeds 7, 21, 25, 73, and 99. Saved mean coverage is 0.9076 for all three methods. Across-run standard deviations are 0.0154, 0.0074, and 0.0101 for collapsed, radial, and strip, respectively. Mean set sizes are 1.0296, 1.0812, and 0.9636, and mean empty-set rates are 0.0800, 0.0776, and 0.0880. Strip's smaller mean set size must therefore be interpreted alongside its empty sets. Five repetitions support an illustration of variability, not a precise or general ranking.

### 7.3 Iris: from measurements to predictor scores

Iris is used as a small guided example [5]. The notebook separates 60 flowers for fitting base classifiers, 60 for conformal fitting, and 30 for final testing. Each species contributes ten S1 and ten S2 observations. Logistic regression, five-nearest-neighbor classification, and Gaussian naive Bayes produce three scores for each candidate species. Preprocessing is fitted within the base-training stage, and classifier probability columns are matched using the fitted class ordering.

The remote revision uses seed 25 and \(\alpha=0.10\). Saved results cover 24/30 observations for collapsed, 25/30 for radial, and 23/30 for strip. All three report a mean set size of approximately 0.8667. With ten calibration observations per class, the 90% rank is the maximum calibration score. The small calibration and test sizes make this an instructional example rather than a reliable benchmark for superiority or a robust audit of coverage.

These values differ from earlier local runs using other seeds or \(\alpha\). A paper must pair every result with its exact configuration; notebook filenames alone do not identify an experiment.

### 7.4 Football-player roles

The remote repository includes a larger example based on FIFA 19 player attributes. The notebook attributes the source to Karan Gadiya's dataset and pins a public mirror by commit and checksum. It reports 18,147 retained players from 18,207 source records. The target groups recorded positions into Defender, Midfielder, Attacker, and Goalkeeper. Individual skill ratings supply inputs, while recorded position, positional suitability summaries, and identity fields are excluded from predictors.

Four classifiers generate candidate scores: logistic regression, random forest, histogram gradient boosting, and a probability-calibrated support-vector classifier. The declared data allocation is 40% base training, 20% validation, 30% conformal fitting, and 10% final testing. The score table contains four predictors for each of four candidate classes. A separate probability-calibration step inside base training must not be confused with conformal S2 calibration.

On 1,815 test players, the saved output reports 1,616 covered observations for collapsed and 1,615 each for radial and strip. Corresponding mean set sizes are 1.0419, 1.0474, and 1.0579. These near-equal results do not support a claim that the geometric methods outperform averaging. Moreover, FIFA attributes are subjective game assessments related to recorded roles; this experiment evaluates reconstruction of those labels, not prospective scouting quality or real-world sporting performance.

**Author note — dataset:** independently verify retained-row criteria, reuse terms, source checksum, and saved-output reproducibility. The notebook's source attribution is available, but the original dataset page was not independently retrieved during this drafting pass. The README also describes a 27-position example not found in the inspected tree; it is not counted as completed evidence here.

### 7.5 Phage-host classification: motivating historical study

The thesis [6] reports a phage score matrix containing 15,638 observations, 54 host labels, and 2,036 host-function score columns. Host-specific vectors contain 26–40 columns. Its protocol uses 9,101 fitting observations, 2,593 validation observations, and 3,944 test observations.

Under that earlier implementation and protocol, reported test coverage and mean set size were 0.929 and 6.09 for collapsed, 0.939 and 15.25 for radial, and 0.905 and 15.35 for strip. Those values are historical thesis results, not measurements of the reviewed library release. The implementations differ in shape quantiles, monotonic smoothing, small-sample calibration, and other details, and some workflows apply Gram filtering and forced predictions.

A prior local audit of an earlier library state found substantially lower strip coverage on validation split 1. It cannot be combined with the thesis test results as if only the algorithm changed: data split, configuration, and implementation provenance differ. The biological experiment requires a fresh versioned reproduction, including base-predictor provenance and the role of genome/protein clusters. No biological claim of reliable nominal coverage is made for the current release in this draft.

## 8. Discussion

The examples illustrate why coverage validity and predictive efficiency should be separated. A valid rank calibration can be applied to many fixed scores, including simple averages and complicated geometric constructions. More flexible geometry does not guarantee smaller useful sets. Correlation, score dimension, support in local neighborhoods, and numerical regularization all influence efficiency.

The collapse baseline is especially important. Its normalization cancels in the calibrated acceptance ranking, making it a simple reference rather than a weak method by definition. The football demonstration currently gives it slightly smaller sets than the geometric alternatives at nearly identical observed coverage. Conversely, the synthetic repeated runs suggest a smaller mean set size for strip, but also a higher empty-set rate. These findings motivate a study of regimes rather than a universal winner.

Missing-score handling should be understood as defining a score on a space of values and masks. A rule that ignores unavailable coordinates can remain meaningful without explicit imputation, but this does not remove the need for exchangeability of the resulting calibration and future scores. Neither structural missingness nor out-of-fold score generation automatically supplies a complete proof of that assumption. Changes in predictors, cluster composition, or observed-coordinate patterns can all matter.

Several limits remain. Small classes may force infinite thresholds. Near-zero strip limits can create excessive scaling. Sparse high-dimensional radial neighborhoods may rely mainly on marginal fallback. Fully missing observations are not yet handled by a unified calibration-compatible policy. Per-class feature selection in two methods currently inspects calibration rows. The present evidence also lacks broad comparisons with established aggregation methods and repeated evaluation across sufficiently diverse datasets.

These qualifications identify the next work needed for a strong paper. They do not erase the practical contribution of a common, inspectable implementation. They do mean that the paper should avoid claiming assumption-free coverage, automatic missingness-pattern validity, a new general conformal theorem, or universal empirical superiority.

## 9. Conclusion

Conformal Envelopes provides a common framework for constructing class-specific prediction sets from multiple predictor scores, including partially observed vectors. Observed-coordinate averaging, radial scoring, and pairwise strip constraints express different aggregation choices within the same calibration interface. The examples show how to separate base-model learning from conformal fitting and how to report coverage together with ambiguity and rejection. The formal guarantee rests on exchangeability and a score construction fixed independently of final calibration, while the practical value of each geometry remains an empirical question. A submission-ready evaluation should resolve the identified implementation qualifications, reproduce all reported examples, and compare methods across predeclared data and missingness regimes.

## Data, code, and declarations

**Code:** <https://github.com/Nizalia13/Conformal_Classification_Library>, reviewed revision `e60c4ecbbbc8393be80a23d2468af0eee99c3f42`. License and archival release identifier: to be confirmed.

**Data:** synthetic data are generated by the repository example; Iris is supplied through scikit-learn; football source provenance is recorded in its notebook. Availability and permissible redistribution of the biological data require author confirmation.

**Prior thesis:** this manuscript develops material from Nizalia Siddiqui's September 2026 master's report, *Conformal Set Prediction for Phage Host Classification*. Its relationship to the paper should be disclosed and cited using a repository identifier if available.

**Author contributions, funding, conflicts, acknowledgments, and any journal-required disclosure of writing assistance:** to be supplied or approved by the authors. No declaration is inferred from the thesis acknowledgments.

## References

1. Ochoa Rivera E, Patel Y, Tewari A. *Conformal Prediction for Ensembles: Improving Efficiency via Score-Based Aggregation*. arXiv:2405.16246, version 3, 2025. <https://arxiv.org/abs/2405.16246>. Verify the final proceedings citation before submission rather than carrying over the thesis's venue/year uncritically.
2. Angelopoulos AN, Bates S. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511, version 6, 2022; published tutorial: *Conformal Prediction: A Gentle Introduction*, Foundations and Trends in Machine Learning 16(4), 494–591, 2023. <https://arxiv.org/abs/2107.07511>.
3. Romano Y, Sesia M, Candès E. *Classification with Valid and Adaptive Coverage*. Advances in Neural Information Processing Systems 33, 2020. <https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html>.
4. Thurin G, Nadjahi K, Boyer C. *Optimal transport-based conformal prediction*. Proceedings of Machine Learning Research 267, 59509–59527, 2025. <https://proceedings.mlr.press/v267/thurin25a.html>.
5. Scikit-learn developers. *load_iris: dataset documentation*. Accessed 28 September 2026. <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html>.
6. Siddiqui N. *Conformal Set Prediction for Phage Host Classification*. Master's report, Master of Science in Industrial and Applied Mathematics, September 2026. Local source: `Internship Report/main.pdf`; public archival citation to be supplied.
7. Gadiya K. *FIFA 19 dataset*. Source named in the repository notebook: <https://www.kaggle.com/datasets/karangadiya/fifa19>. Metadata and reuse terms require verification. Pinned mirror: <https://github.com/qualityjacks/Fifa19_Insights/blob/334610dd4516e0471db53a47bb195a35c808f926/data.csv>.
