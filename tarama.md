# Winning a Turkish Query–Product Relevance Competition: A Literature Review and Engineering Playbook

## TL;DR
- **The competition is fundamentally a Positive-Unlabeled (PU) + negative-sampling problem, and the winner will be decided by negative-generation quality and macro-F1 threshold calibration far more than by model choice.** Treat constructed pairs as negatives (biased PU), but generate negatives as a *blend* dominated by easy/random with a thin slice of hard negatives (Facebook EBR found improvement "continued up to a 100:1 easy-to-hard ratio"), mine hard negatives from rank 101–500 rather than the absolute top, and filter false negatives at ≤95% of the positive score (NV-Retriever).
- **Build a two-track ensemble**: (1) a Turkish/multilingual cross-encoder relevance classifier (use `Trendyol/TY-ecomm-embed-multilingual-base` for retrieval/bi-encoder features and BERTurk or a multilingual XLM-R/mDeBERTa cross-encoder for scoring) and (2) a GBDT (LightGBM/CatBoost) over lexical + structural features (BM25, token coverage, category/gender/age consistency, attribute key:value matches). Blend their scores, then tune a single decision threshold to maximize macro-F1.
- **Validation must mimic the mixed test distribution, not the all-positive train.** Construct a held-out validation set with synthetic negatives at the estimated test prevalence, probe the public leaderboard with all-ones/all-zeros submissions to back out the positive rate, and lock the threshold against that estimate to avoid private-leaderboard shake-up.

## Key Findings
1. **PU learning**: With only positives, the practical routes are (a) biased PU (sample negatives, treat as label 0) and (b) nnPU (non-negative risk estimator) if you can estimate the class prior π. Two-step "spy" methods (S-EM) reliably identify trustworthy negatives. For IR/matching at competition scale, biased PU with engineered negative sampling dominates.
2. **Negative sampling is the crux**: random vs in-batch vs hard negatives each play a role. Hard negatives *alone* hurt; the proven recipe is mostly-random with a thin slice of hard negatives, sourced from mid-rank candidates, with explicit false-negative filtering.
3. **e-commerce relevance architectures**: Industry (Amazon ESCI winners, Alibaba/JD, Walmart, Etsy, Facebook) converges on bi-encoder retrieval + cross-encoder reranking, concatenating structured fields (brand, color, category) into the text input. The KDD Cup 2022 winner used a cross-encoder over `[CLS]query[SEP]color:<c> brand:<b> description:<title+bullets>[SEP]`.
4. **Turkish NLP**: Best options are the Trendyol e-commerce embedding model, BERTurk, turkish-e5-large, and BGE-M3 / multilingual-e5-large. Turkish morphology (agglutination) and diacritics/ASCII-folding ("şapka" vs "sapka") demand normalization; Zemberek and SymSpell handle stemming/typos.
5. **Macro-F1 optimization**: Macro-F1 is prevalence-sensitive and not optimized by accuracy/AUC; threshold tuning on a distribution-matched validation set is essential, and 0.5 is the theoretical upper bound on the optimal F1 threshold.
6. **GBDT vs neural & ensembling**: GBDTs dominate tabular/lexical features; cross-encoders dominate semantic matching. Blending the two is the standard winning move.
7. **Kaggle CV discipline**: With train (all-positive) ≠ test (mixed), build a CV scheme that recreates the test mixture; use limited submissions (5/day) for leaderboard probing, not threshold hill-climbing.

## Details

### 1. Positive-Unlabeled (PU) Learning

The training data here is the classic **case-control PU scenario**: positives are sampled separately, and you must treat the rest of the (query, product) space as unlabeled. The Bekker & Davis survey (arXiv:1811.04820) groups methods into three families:

- **Biased PU / "unlabeled as negative."** Treat constructed pairs as negatives and train a standard classifier, accepting label noise. This is the workhorse for competitions because it is simple and pairs directly with sophisticated negative sampling. The noise rate is controlled by how cleanly you generate negatives.
- **Class-prior incorporation & risk estimators.** **Elkan & Noto (2008)** showed a probabilistic classifier trained on PU data classifies positives as positive with constant probability c; ĉ (mean predicted probability on held-out positives) estimates label frequency and lets you rescale scores. **nnPU (Kiryo et al., NeurIPS 2017, arXiv:1703.00593)** fixes the overfitting of unbiased PU (uPU) by clamping the negative-class empirical risk at zero — essential when using flexible neural nets with limited positives. nnPU requires a known class prior π = P(y=1); a documented weakness is that its negative-risk estimator can approach 0, "over-playing" the negative class and producing imbalanced error rates (NeurIPS 2025 poster on balancing PU error rates) — directly relevant because macro-F1 punishes class imbalance.
- **Two-step methods.** Step 1: identify reliable negatives; Step 2: train a supervised classifier on positives + reliable negatives. **S-EM (Liu et al., 2002)** uses the **"spy" technique**: inject a small fraction (S-EM default ~15%) of positives as "spies" into the unlabeled set, train a Naïve Bayes classifier, and set the reliable-negative threshold so (almost) all spies are recovered as positive; anything scoring below is a reliable negative. PEBL (1-DNF) and Roc-SVM (Rocchio) are alternatives. These remain strong baselines two decades later.

**Class-prior estimation** (needed for nnPU and for calibrating macro-F1): Elkan-Noto's calibration, du Plessis–Sugiyama Pearson/penalized-f-divergence minimization, and TIcE are the standard tools. Note Blanchard et al. and Jain et al. proved prior estimation is ill-posed without an "irreducibility"/non-overlap assumption.

**Recommendation for this competition**: Use **biased PU with engineered negatives** as the primary approach (it lets you exploit the structured catalog), and use Elkan-Noto-style calibration plus leaderboard probing to pin down π for threshold setting. Keep nnPU as a secondary neural variant if you can estimate π; it is most valuable if your negatives are noisy.

### 2. Negative Sampling / Hard-Negative Mining

This is where the competition will be won or lost. The data gives you positives only; the *distribution* and *quality* of generated negatives directly determine calibration and the optimal macro-F1 threshold.

**The three negative types**:
- **Random/easy negatives**: random products from the catalog. Cheap, necessary, but too easy alone — the model never learns a sharp boundary.
- **In-batch negatives**: other queries' positives within a training batch (used by DPR; RocketQA pushes batch size up to thousands). Efficient for bi-encoders since embeddings are already computed. sentence-transformers' `MultipleNegativesRankingLoss` defaults to ~4 in-batch negatives.
- **Hard negatives**: products that look relevant but are not. Sources: **BM25** (lexical), **ANN/embedding** (ANCE-style dynamic mining; STAR static hard negatives + random for stability), **taxonomy-based** (sibling categories), and **behavioral** (impressed-not-clicked — not available here).

**The critical empirical findings** (from industry, verbatim where load-bearing):
- **Facebook EBR (Huang et al., KDD 2020, arXiv:2006.11632)**: "models trained simply using hard negatives cannot outperform models trained with random negatives," and "Increasing the ratio of easy-to-hard negatives continued the model improvement up to a 100:1 easy-to-hard ratio." The best hard negatives are mid-rank, not the top: "using the hardest examples is not the best strategy… sampling between rank 101-500 achieved the best model recall." Using non-click impressions alone as negatives produced an "absolute 55% regression in recall for people embedding model." Offline hard-negative mining added ~3.4% recall in a cascade.
- **The false-negative problem**: per RocketQA's audit (via NV-Retriever), "about 70% of passages most similar to the queries should be actually labeled as positive." This is acute here — the catalog likely contains many unlabeled-but-relevant products per query, so naive top-k mining will be dominated by false negatives.
- **Mitigations**:
  - **NV-Retriever (NVIDIA, 2024, arXiv:2407.15831)** "positive-aware" filtering (TopK-PercPos): "set the maximum threshold for the negative relevance score as 95% of the positive score" — found to be "the optimum configuration." Variants: absolute cutoff 0.7 (TopK-Abs) or "subtracting a small margin (0.05) from the positive score" (TopK-MarginPos). Best sampling: take 4 negatives from the top-10 after filtering. NV-Retriever-v1 reached 60.9 NDCG@10 on the MTEB/BEIR Retrieval track, ranking 1st in July 2024.
  - **RocketQA denoising**: train a cross-encoder, discard mined negatives it scores as confidently positive; a common implementation keeps cross-encoder scores in **[0.1, 0.9]** and samples 4.
  - **Taxonomy-based hard-negative sampling (TB-HNS, arXiv:2511.00694)**: sample negatives from the *sibling* category one level up in the hierarchy, excluding any overlap with the positive set. It "consistently outperforms random, BM25, and ANCE mining across Recall@k" on ESCI and is cheap — perfectly suited to the `category` chain in items.csv.

**Positive:negative ratios**: A ~**1:4** positive:negative ratio (Amazon Semantic Product Search style) is a sound baseline; bi-encoder training uses far more via in-batch negatives. sentence-transformers defaults to **3–5 hard negatives per anchor** when mining and **~4 in-batch negatives** for MultipleNegativesRankingLoss. Cross-encoders are "particularly receptive to strong hard negatives" but "if you only use hard negatives, your model may unexpectedly perform worse for easier tasks" — echoing the Facebook finding.

**Negative-generation strategy for this competition** (concrete):
1. **Easy negatives**: random products, mostly filtered to differ from the positive's category/gender/age — but keep some that violate these as "obvious" negatives.
2. **Hard negatives**: (a) **taxonomy-based** (same parent, different leaf category); (b) **BM25 mid-rank** (rank ~50–300 on the title field); (c) **embedding ANN** from the Trendyol model, mid-rank (rank 101–500).
3. **Filter false negatives**: drop any candidate scoring above 95% of the query's positive similarity, or with a cross-encoder score >0.9.
4. **Blend** strongly toward easy negatives (the 100:1 finding) for a bi-encoder; for a cross-encoder, a milder 4:1 to 10:1 easy:hard works because cross-encoders benefit from harder examples — but validate against macro-F1.

Because **negative quality directly shifts calibration**, the negative mix should approximate the test mix; otherwise your decision threshold will be biased.

### 3. Query–Product / e-commerce Semantic Matching Architectures

**Bi-encoder vs cross-encoder**: Bi-encoders (two-tower, Siamese) encode query and product independently → fast, cacheable, ideal for first-stage retrieval. Cross-encoders jointly encode the pair with full cross-attention → consistently 5–10 nDCG points more accurate but too slow for retrieval; they belong in the second-stage reranker. For a **binary relevance classification** task scored per supplied pair (not a retrieval task over the whole catalog), the **cross-encoder is the natural primary model** since you only score the given pairs.

**Incorporating structured attributes**: The dominant pattern is to **serialize structured fields into the text input**. The KDD Cup 2022 winner (NetEase team "www", arXiv:2208.02958) used `[CLS]query[SEP]color:<color> brand:<brand> description:<title+bullet_point+description>[SEP]` fed to DeBERTa/XLM-R/RemBERT cross-encoders, trained as 4-class ESCI classification with cross-entropy, then collapsed to a score via weighted sum of class probabilities (P_E + 0.1·P_S + 0.01·P_C), achieving NDCG 0.9043 for 1st place. They added: typed entity markers, translation-based data augmentation, adversarial training (AWP/FGM), self-distillation, pseudo-labeling, label smoothing, and ensembling. Other top solutions used similar cross-encoder + heavy-ensemble recipes.

**Industry systems**: Amazon Semantic Product Search (Nigam et al., KDD 2019) uses a 3-part hinge loss separating purchased, impressed-not-purchased, and random negatives. Walmart (2024, arXiv:2408.04884) uses stratified sampling + in-batch + offline hard negatives + 50% typo augmentation. Etsy (arXiv:2306.04833) combines hard in-batch + uniform negatives (uniform helps bottom-ranking, hard helps top). JD/Alibaba (Graph-based Multilingual Product Retrieval, arXiv:2105.02978) define behavior-based, offline-model-based, and online-model-based hard negatives. Amazon also showed a high-precision cross-encoder used as a re-ranking feature and training objective (ECNLP 2022).

For this competition, **also engineer structural consistency signals** (Section 6) since the catalog gives clean category/brand/gender/age fields.

### 4. Turkish-language NLP & Multilingual Text Matching

**Best pretrained models (2025–2026)**:
- **`Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0`** — a SentenceTransformer (~768-dim, max 384 tokens, Matryoshka-truncatable) distilled from gte-multilingual-base and fine-tuned on e-commerce datasets plus millions of real Turkish queries + product interactions. This is the single most domain-aligned model available and should anchor your bi-encoder/retrieval features and ANN hard-negative mining. (Caveat from its own model card: "Semantic similarity may incorrectly assign high similarity scores to unrelated but lexically similar or frequently co-occurring phrases.")
- **BERTurk** (`dbmdz/bert-base-turkish-cased`) — the de facto Turkish monolingual BERT and a strong base for a fine-tuned cross-encoder.
- **turkish-e5-large** (ytu-ce-cosmos) — E5 fine-tuned on Turkish corpora.
- **BGE-M3** — XLM-R-based, supports dense + sparse + multi-vector in one model (useful for hybrid lexical+semantic), 100+ languages, up to 8192 tokens. A Turkish-finetuned BGE-M3 exists.
- **multilingual-e5-large / -instruct** — TR-MTEB evaluations and a Turkish production blog (MDP Group) found mE5-large "the most stable and reliable" across Turkish tasks, with threshold stability around 0.89–0.92.
- **LaBSE** — solid multilingual baseline but generally superseded by BGE-M3/E5.

**Turkish-specific handling**:
- **Diacritics / ASCII-folding**: Turkish users frequently omit diacritics ("şapka"→"sapka", "ç"→"c"). Naive ASCIIfication creates invalid words or changes meaning (homographic ambiguity), so prefer **deASCIIfication** (restoring accents, à la Yüret's deasciifier) over stripping; research (Information Processing & Management) shows "diacritics restoration approach yielded more effective and robust results compared with normalizing tokens to remove diacritics."
- **Agglutination/morphology**: suffixes explode the vocabulary. Use **Zemberek** (morphological analysis, stemming, normalization) or Snowball; fixed-length truncation stemming (first 4–5 chars) is a cheap, surprisingly effective alternative for Turkish IR.
- **Typos**: SymSpell + Levenshtein for query typo correction (used by Trendyol competition finalists), preserving brand names.
- **Lexical features**: character n-grams (robust to suffix variation), TF-IDF, BM25 on normalized text.

**The Trendyol/TEKNOFEST 2025 competition** (the likely origin of this task) reported in Trendyol Tech's Feb-2026 writeup that finalists converged on: **hybrid search (BM25 + embeddings)** as "the new default," Turkish-first query understanding (domain-aware typo correction preserving brand names), multi-stage ranking pipelines, **CatBoost** ("the absolute choice," with YetiRank/QuerySoftMax) and **LightGBM** (LambdaRank) for ranking, Polars + Parquet for data (used by all 20 finalist teams), FAISS for ANN, the Trendyol embedding model, Zemberek/SymSpell for Turkish NLP, and Optuna for tuning.

### 5. Macro-F1 Optimization & Threshold Tuning

**Why macro-F1 is special**: It is the unweighted mean of per-class F1 (relevant and irrelevant weighted equally), so the minority class matters as much as the majority. It is **not** optimized by maximizing accuracy or AUC, and crucially it is **prevalence-sensitive**: changing the test class mix changes macro-F1 even if per-class performance is unchanged (Opitz, 2024). This means your **validation prevalence must match the test prevalence** or your tuned threshold will be wrong.

**Threshold selection**:
- There is no closed-form optimal F1 threshold, but **0.5 is the proven upper bound** on the optimal F1 threshold for a calibrated classifier; the optimum is typically below 0.5 for the rarer class. A reasonable heuristic for the minority class is (P(y=1)+0.5)/2.
- Sweep thresholds on a distribution-matched validation set and pick the one maximizing macro-F1 (linear-time per-class methods exist).
- For two classes you have essentially one threshold; tune it carefully. Per-segment thresholds can add a few points of F1 in multilingual tasks but risk overfitting on small segments.

**Estimating test prevalence via leaderboard probing**: Submit **all-ones** and **all-zeros** and back out the positive rate. With a known test size and the macro-F1 formula, an all-positive submission's score pins down the true positive count (for all-ones: FN=0, so F1_pos = 2·TP/(2·TP+FP) = 2·P/(1+P) where P is the true positive fraction, and F1_neg = 0; macro-F1 = ½·F1_pos lets you solve for P). This is a legitimate, high-value use of 1–2 daily submissions. Beware: probing leaks information about the *public* split only, and over-probing risks overfitting the public set (the "Ladder"/wacky-boosting cautionary results of Blum & Hardt).

**Pitfalls**: Don't tune the threshold on the public leaderboard (overfits the public split → private shake-up). Tune on local validation; use the leaderboard only to validate the prevalence estimate.

### 6. GBDT vs Neural & Ensembling

**GBDTs (LightGBM/CatBoost/XGBoost) remain state-of-the-art for tabular/lexical features** and frequently beat deep nets on structured data with less tuning and compute (Shwartz-Ziv & Armon, "Tabular Data: Deep Learning Is Not All You Need"; Grinsztajn et al.). CatBoost is especially apt here (native categorical handling for brand/category/gender/age). But GBDTs cannot capture deep semantics — that is the cross-encoder's job.

**Feature engineering for the GBDT track**:
- **Lexical overlap**: token overlap count, **Jaccard**, query-token **coverage** (fraction of query tokens in title), ordered n-gram match.
- **BM25 / TF-IDF cosine** between query and title/attributes (on normalized + deASCIIfied + stemmed text).
- **Embedding cosine** similarity (from Trendyol model / BGE-M3) — feed the dense score as a GBDT feature.
- **Structural consistency**: does the product's `gender` contradict a gendered query? `age_group` consistency? Is the query's implied category present in the product's `category` chain? Brand-in-query match.
- **Attribute key:value matching**: parse the `attributes` string ("materyal: tekstil, renk: …") and count query terms matching attribute values (color, material).
- **Char-n-gram similarity** (robust to Turkish suffixes/typos), lengths, OOV rates, brand-presence flags.

**Ensembling**: The winning move is to **blend GBDT (lexical/structural) scores with cross-encoder (semantic) scores** — they capture complementary signals. Options: weighted average, stacking (a meta-learner over both), or feeding the cross-encoder score as a GBDT feature. Diversify across model classes and seeds to reduce variance and shake-up risk. The KDD Cup 2022 winner relied heavily on ensembling multiple cross-encoder backbones; "an ensemble of deep models and XGBoost" outperforming XGBoost alone is a well-documented pattern.

### 7. Kaggle Best Practices for This Setup

- **CV that mimics test**: The train set is all-positive; test is mixed. Build validation folds by holding out some positives AND injecting synthetic negatives (using your negative-generation pipeline) at the **estimated test positive rate**. Group by `term_id` (or query) so the same query doesn't leak across folds — this prevents over-optimistic CV.
- **Correlate CV with LB**: Track local macro-F1 vs public LB on every submission; if they diverge, your negative distribution or prevalence assumption is off.
- **Avoid shake-up**: Don't pick final submissions purely by public LB ("the public leaderboard is not your friend"). Choose from multiple diverse models; keep a robust, well-CV'd submission as one of your two final picks.
- **Budget the 5 submissions/day**: Spend 1–2 early submissions on prevalence probing (all-ones/all-zeros), then use the rest to confirm CV↔LB correlation, not to hill-climb the threshold.
- **Tooling**: Polars + Parquet for fast data wrangling (Trendyol finalists' choice), FAISS for ANN hard-negative mining, Optuna for tuning, CatBoost/LightGBM for the GBDT track, sentence-transformers `mine_hard_negatives()` (with `range_min`, `max_score`, `relative_margin` to implement NV-Retriever-style filtering) for the neural track.

## Recommendations

**Stage 1 — Baseline (Days 1–2)**:
1. Normalize Turkish text (lowercase with Turkish casing rules, deASCIIfication, optional Zemberek stemming; SymSpell typo correction preserving brands).
2. Generate negatives: per positive query, sample ~10 random + 2–3 taxonomy-based (sibling category) + 2–3 BM25/embedding mid-rank (rank 101–500) negatives; filter any scoring >95% of the positive similarity.
3. Train a LightGBM/CatBoost on lexical + structural + embedding-cosine features. Probe the leaderboard (all-ones/all-zeros) to estimate test prevalence. Tune the threshold on distribution-matched validation. **Benchmark: this should give a respectable macro-F1 and a reliable CV↔LB link.**

**Stage 2 — Semantic model (Days 3–6)**:
4. Fine-tune a cross-encoder (BERTurk or a multilingual XLM-R/mDeBERTa) on the same positives + mined negatives, serializing structured fields into the input (KDD'22 format: `query [SEP] brand:… category:… attributes:… title`). Use the Trendyol embedding model for bi-encoder features and ANN hard-negative mining.
5. Add false-negative denoising (cross-encoder score ∈ [0.1, 0.9]; relabel confident positives).

**Stage 3 — Ensemble & calibrate (Days 7+)**:
6. Blend GBDT + cross-encoder scores (start with simple weighted average, then stack). Re-tune the single decision threshold on distribution-matched validation to maximize macro-F1.
7. Diversify: multiple backbones/seeds. Keep two final submissions — one max-CV ensemble, one robust single model.

**Thresholds/benchmarks that change the plan**:
- If CV macro-F1 ≫ public LB: your negative mix or prevalence estimate is wrong — re-probe and re-balance negatives.
- If hard negatives degrade validation macro-F1: dial back toward the 100:1 easy:hard ratio (Facebook EBR).
- If the cross-encoder overfits fast (it will — cross-encoders "overfit rather quickly"): use an evaluator with load-best-model and strong early stopping.
- If π (positive prevalence) is far from 50%: lower the threshold for the minority class and consider class weights / nnPU.
- If a single cross-encoder already beats the GBDT+blend on CV: still keep the GBDT in the ensemble for diversity/robustness against shake-up.

## Caveats
- **The exact test positive prevalence is unknown** until you probe the leaderboard; all threshold guidance is contingent on that estimate. Because macro-F1 is prevalence-sensitive, this is the highest-leverage unknown.
- **The ~1:4 positive:negative figure for Amazon Semantic Product Search is from a secondary survey, not verbatim from the paper** — treat it as an approximate starting point, not a rule.
- **False negatives are likely rampant** given a large catalog with many relevant products per query (RocketQA found ~70% of naive top-k "negatives" were actually relevant); without denoising, mined "negatives" will corrupt training and bias calibration.
- **The Trendyol embedding model is capped at 384 tokens** and, per its model card, "may incorrectly assign high similarity scores to unrelated but lexically similar … phrases" — validate, don't trust blindly.
- **KDD Cup 2022 ESCI was a 4-class ranking (NDCG) task with provided labels**, not a PU binary task; its architecture transfers but its negative-sampling-free setup does not — you still must generate negatives.
- **Leaderboard probing leaks only the public split**; over-probing or threshold hill-climbing on public LB risks private-leaderboard shake-up.
- Several cited blog/vendor sources (Medium, Weaviate, MDP Group) corroborate but are secondary; the core claims are anchored in peer-reviewed papers (Facebook EBR, RocketQA, NV-Retriever, Kiryo et al., Liu et al., the KDD Cup 2022 winning paper) and primary model cards where possible.