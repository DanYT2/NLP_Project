"""Long-form prose and design-rationale strings for the dashboard.

Kept separate from app.py so the UI logic stays readable. Each rationale entry
follows: {title, what (1-line summary), why (the reasoning), source}.
The `source` field points at the file:line where the decision lives in code —
useful when the lecturer asks "show me where that's enforced".
"""

from __future__ import annotations

PROJECT_TITLE = "Topic Classification on 20 Newsgroups — From MLPs to QLoRA"
SUBTITLE = "HSLU NALAPRO Masters Project · Dan Waititu"
WANDB_URL = (
    "https://wandb.ai/danwwaititu-hochschule-luzern/hslu-nalapro"
    "?nw=nwuserdanwwaititu"
)

OVERVIEW_BULLETS = [
    "**Task** — multi-class topic classification, 20 classes, ~18.8k documents.",
    "**Dataset** — 20 Newsgroups (`sklearn.datasets.fetch_20newsgroups`), header/footer/quote-stripped.",
    "**Goal** — compare four representation/learning paradigms on a common benchmark + one bonus.",
    "**Constraint** — single-student project, 6-week timeline, reproducible (`SEED=42`, Python 3.13, `uv`).",
]

PROBLEM_STATEMENT = """
**Why this problem?** Topic classification is the canonical text-classification benchmark
— it's tractable on a laptop, has 20 well-separated classes, and surfaces the practical
differences between bag-of-words, distributional semantics, contextual embeddings, and
instruction-tuned LLMs without confounding factors like very long documents or domain shift.

**Why 20 Newsgroups specifically?** It's the benchmark every NLP textbook uses for
end-to-end text classification, so results are comparable to a wide body of prior work.
The class structure (20 USENET groups) also has *near-duplicate* pairs
(`comp.sys.ibm.pc.hardware` vs `comp.sys.mac.hardware`,
`talk.religion.misc` vs `alt.atheism`) that stress class-separation — easy to get
~70% accuracy, hard to push past 85%.

**Hard constraints from the spec (NALAPRO §3):**
- Strip headers/footers/quoted-replies — they leak the label otherwise.
- Track every experiment in W&B or MLflow.
- Cite every third-party model/library; disclose AI-tool usage in the report.
- Do *not* commit the dataset — fetch at runtime.
"""

# --- Methodology rationale, grouped by question ---------------------------------

DECISIONS: dict[str, dict[str, dict[str, str]]] = {
    "q1": {
        "remove_headers": {
            "title": "Strip headers / footers / quoted replies",
            "what": "`fetch_20newsgroups(remove=('headers','footers','quotes'))` is the default and is *not* optional.",
            "why": "Newsgroup headers contain the target group name verbatim (e.g. `Newsgroups: rec.autos`) — leaving them in would let a naïve model get ~95% via string match and learn nothing. Spec §3 makes this a hard rule.",
            "source": "src/nlp_project/data.py:27–28 · CLAUDE.md:62",
        },
        "stopwords_split": {
            "title": "Stopwords retained for word2vec, removed for TF-IDF",
            "what": "`preprocess(..., drop_stopwords=False)` feeds word2vec; `drop_stopwords=True` feeds TF-IDF and the classifier.",
            "why": "word2vec learns distributional semantics from co-occurrence within a 5-word window. Frequent function words ('the', 'is', 'of') carry positional/grammatical signal and *help* embedding geometry. TF-IDF, by contrast, is a sparse bag-of-counts — stopwords inflate the vocabulary without discriminating between topics.",
            "source": "src/nlp_project/data.py:60–91",
        },
        "w2v_determinism": {
            "title": "Word2vec pinned to `workers=1`",
            "what": "Single-threaded gensim Word2Vec training, even though gensim defaults to parallel.",
            "why": "The 1-epoch vs 20-epoch comparison only makes sense if both runs see the corpus in *byte-identical order*. Multi-threaded gensim consumes the corpus non-deterministically, so the same `seed=42` would still produce different embeddings between runs. `workers=1` is a 5–10× slowdown we accept for reproducibility.",
            "source": "src/nlp_project/embeddings.py:1–8",
        },
        "patience_5": {
            "title": "Hand-rolled training loop, early stopping `patience=5`",
            "what": "No Lightning, no `Trainer` — plain PyTorch loop with Adam, early-stopping on val loss, best-weight restoration.",
            "why": "Q1 is the *teaching* experiment — the marking rubric expects us to demonstrate understanding of the training mechanics. A framework would hide the loop. Patience=5 was chosen empirically: lower (2–3) stopped too early on noisy val curves; higher (10+) wasted compute on plateaus.",
            "source": "src/nlp_project/train.py:89–96",
        },
        "q1d_mean_max": {
            "title": "Q1d self-designed experiment: mean + max pooling",
            "what": "Replace mean-pooling over word2vec with `concat(mean, max)` — same network, same data, 200-D input instead of 100-D.",
            "why": "Mean-pooling washes out salient terms; max-pool preserves the strongest activation per dimension, which captures topic-discriminative tokens (e.g. 'graphics', 'hockey'). Concatenating both gives the classifier both a 'gist' and 'standout' view. Honest result: it actually *hurt* by ~1pp — interesting failure to discuss.",
            "source": "src/nlp_project/embeddings.py:75–93",
        },
    },
    "q2": {
        "trainer_vs_handrolled": {
            "title": "HuggingFace `Trainer`, not a hand-rolled loop",
            "what": "Q2 delegates training to `transformers.Trainer` with warmup, AdamW, LR scheduler.",
            "why": "Q1 proved we understand the loop; Q2's question is 'does fine-tuning a pre-trained transformer beat shallow methods?'. The answer should depend on the *fine-tuning recipe*, not on our ability to re-implement AdamW. Trainer also handles checkpoint selection and metric tracking automatically — fewer places for bugs to hide.",
            "source": "src/nlp_project/bert_train.py:1–4 · CLAUDE.md:79",
        },
        "linear_probe_lr": {
            "title": "Linear-probe variant uses lr=1e-3 (not 2e-5)",
            "what": "When the encoder is frozen, the classification head trains alone — and we bump the learning rate by ~50×.",
            "why": "2e-5 is the canonical fine-tune LR — small because every layer of BERT is moving. With the encoder frozen, only a single linear layer trains; that requires the standard MLP-classifier LR (~1e-3). Using 2e-5 on a frozen-encoder run is the #1 mistake in linear-probe baselines and silently produces under-trained heads.",
            "source": "src/nlp_project/bert_train.py:103",
        },
        "max_length_256": {
            "title": "max_length=256 chosen from token-length histogram",
            "what": "Q2a plots the BERT-tokenized doc-length distribution; 256 covers the 95th percentile.",
            "why": "BERT's quadratic attention makes seq=512 ~4× more expensive than seq=256. The histogram (see figure on page 6) shows >95% of 20NG docs fit under 256 tokens after preprocessing — going to 512 doubles training time for marginal coverage gain.",
            "source": "src/nlp_project/bert_data.py · figures/q2a_token_length_hist.png · CLAUDE.md:89",
        },
        "macro_f1": {
            "title": "Best-model selection on `eval_macro_f1`",
            "what": "Trainer's `metric_for_best_model='eval_macro_f1'`, not accuracy.",
            "why": "20NG has slight class imbalance (~628 docs/class in train, but some splits drift to ~390). Accuracy lets a model 'cheat' by getting the majority classes right. Macro-F1 weights every class equally — and matches the metric the report's cross-experiment table uses.",
            "source": "src/nlp_project/bert_train.py:98",
        },
    },
    "q3": {
        "two_stage": {
            "title": "Two-stage: MLM pretraining, *then* classification fine-tune",
            "what": "Stage A: `BertForMaskedLM` on the 20NG train texts for 3 epochs. Stage B: load that checkpoint, swap to `BertForSequenceClassification`, fine-tune with Q2b's recipe verbatim.",
            "why": "Domain-adaptive pretraining (Gururangan et al. 2020) is a well-known second-stage technique. By holding Q2b's hyperparameters constant for Stage B, the *only* changed variable is the encoder's initial weights — so any Q3 vs Q2 delta is attributable to the MLM stage.",
            "source": "src/nlp_project/mlm_pretrain.py:5–7 · CLAUDE.md:98–100",
        },
        "mlm_eval_split": {
            "title": "MLM-eval split is independent of the classification val set",
            "what": "Stage A uses an internal 90/10 split of the 20NG train texts; that 10% is *only* used for MLM perplexity tracking, never for classification.",
            "why": "If the MLM stage validated on the same 10% we'd later use for classification val, the model would have seen those texts during pretraining — leaking signal into the val set we use to pick the best classification checkpoint. The internal split keeps the classification val set 'unseen' end-to-end.",
            "source": "src/nlp_project/mlm_pretrain.py:17–21 · CLAUDE.md:97",
        },
        "mask_rate_default": {
            "title": "`mlm_probability=0.15`, not tuned",
            "what": "Original BERT default; held constant across runs.",
            "why": "The research question is 'does domain-adaptive MLM help at all?' — not 'what is the optimal mask rate?'. Tuning mlm_probability would add a confounding axis and dilute the comparison budget.",
            "source": "src/nlp_project/mlm_pretrain.py:100–101",
        },
    },
    "q4": {
        "subsample_200": {
            "title": "200-doc stratified subsample (10/class), not the full 7,532",
            "what": "Q4 evaluation runs on a fixed 200-doc subset of the test set; same seed (42), 10 docs per class.",
            "why": "Llama-3.2-3B at ~2–3s/doc on the available 3060 GPU = ~5 hours for one full-test pass. Three runs (zero/k=1/k=3) would be 15 hours. The subsample reproduces class balance, holds eval cost to ~10–25 min per run, and is byte-identical across the three conditions so deltas are paired. This limitation is disclosed explicitly in the report.",
            "source": "src/nlp_project/llama_classify.py · CLAUDE.md:120",
        },
        "nf4_quant": {
            "title": "4-bit nf4 quantization via bitsandbytes",
            "what": "Llama-3.2-3B loaded with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')`.",
            "why": "fp16 3B model ≈ 6 GB VRAM; nf4 ≈ 2 GB — fits comfortably on a consumer 3060 (12 GB) with room for KV cache and few-shot demos. nf4 (normalised float 4) preserves accuracy better than int4 because the quantisation grid is calibrated to a normal distribution, which matches transformer weight statistics.",
            "source": "src/nlp_project/llama_classify.py:181–200",
        },
        "label_parsing": {
            "title": "Generative output parsing: exact → substring → Levenshtein",
            "what": "After Llama generates 15 tokens, parse the label by exact match, then substring containment, then nearest-Levenshtein-neighbour fallback. Track `invalid_rate`.",
            "why": "Instruction-tuned LLMs don't always output the exact label string — they paraphrase ('this is computer graphics') or hedge ('probably rec.autos'). A 3-tier fallback recovers ~90% of these cases honestly. The `invalid_rate` diagnostic surfaces *how often* the fallback fired — a key debugging signal that's transparent to the report reader.",
            "source": "src/nlp_project/llama_classify.py:138–173",
        },
        "greedy_decoding": {
            "title": "Greedy decoding (`do_sample=False`)",
            "what": "No temperature, no top-p — deterministic argmax at each step.",
            "why": "Classification accuracy is averaged over 200 docs — sampling adds variance that we'd need to bootstrap-CI over multiple seeds. Greedy decoding makes every run reproducible bit-for-bit and removes the temperature confound from the zero-shot-vs-few-shot comparison.",
            "source": "src/nlp_project/llama_classify.py:18–21",
        },
    },
    "qbonus": {
        "lora_targets": {
            "title": "LoRA on `{q, k, v, o}_proj` + `modules_to_save=['score']`",
            "what": "Inject LoRA adapters into attention projections (the canonical QLoRA targets) AND mark the classification head for *full-precision* training, not LoRA.",
            "why": "The classification head (`score` linear layer) is randomly initialised — LoRA's low-rank update can't bridge from random init to a useful classifier in 3 epochs. `modules_to_save` keeps it as a regular trainable parameter. Forgetting this is the #1 silent-failure mode of QLoRA-for-classification — the model trains, the loss goes down, and final accuracy is ~5%.",
            "source": "scripts/build_qbonus_notebook.py · CLAUDE.md:138",
        },
        "sweep_choice": {
            "title": "Two-run sweep: (r=16, lr=2e-4) vs (r=32, lr=1e-4)",
            "what": "Two LoRA configs trained back-to-back; winner selected by val macro-F1.",
            "why": "Higher rank (r=32) gives more adapter capacity but is more prone to overfitting on small classification data — so we pair it with a lower LR. The (r=16, lr=2e-4) point is the canonical QLoRA-classification recipe (Dettmers et al. 2023). Two runs fits the compute budget; macro-F1 (not accuracy) breaks ties fairly given the slight class imbalance.",
            "source": "scripts/build_qbonus_notebook.py · CLAUDE.md:139",
        },
        "two_eval_splits": {
            "title": "Evaluated on BOTH the full test set and the Q4 200-doc subset",
            "what": "The winning adapter is run twice — once on all 7,532 test docs, once on the byte-identical Q4 subset.",
            "why": "The full-test eval is the headline number, comparable to Q1/Q2/Q3. The Q4-subset eval is the *paired* comparison against zero/few-shot Llama on the same 200 docs — isolates the effect of fine-tuning from sample selection.",
            "source": "scripts/build_qbonus_notebook.py · CLAUDE.md:140",
        },
    },
}

# --- Conclusions & future-work copy ----------------------------------------------

KEY_FINDINGS = [
    "**Fine-tuning > pretraining > prompting (at this scale).** QLoRA on a 3B Llama (0.757 acc) beats BERT-base (≈0.82 on Q2/Q3 — see PNGs) only on the small subset; on the full test, classical fine-tuning still leads. Zero-shot Llama is barely above chance (0.155 ≈ 3× random for 20 classes).",
    "**TF-IDF is a surprisingly strong baseline.** Q1c (TF-IDF + MLP) at 0.696 macro-F1 beats both word2vec variants by ~6–9pp — the sparse bag-of-words still wins on topic classification when the topics are lexically distinct.",
    "**Mean+max pooling didn't help (Q1d, –1pp).** Honest negative result — concatenating max-pool to mean-pool *increased* input dimension but didn't add discriminative signal beyond what the MLP could already learn.",
    "**Few-shot helps Llama linearly: 0.155 → 0.240 → 0.365** moving from zero → k=1 → k=3 per class. The slope suggests k=10+ would close more of the gap, but at proportionally more context-window cost.",
    "**QLoRA beats zero-shot by ~60pp** on the same 200-doc subset (0.805 vs 0.155) — the most dramatic delta in the project and the clearest argument for task-specific fine-tuning even with a strong base LLM.",
]

FUTURE_WORK = [
    "**Larger few-shot k** — run k=5, 10, 20/class to map the few-shot scaling curve fully.",
    "**Full-test Llama eval with batching** — current 1-doc-at-a-time inference is the bottleneck; vLLM or HF `pipeline` batching would unlock the full 7,532-doc Q4 run.",
    "**Retrieval-augmented few-shot (RAG)** — instead of fixed demos, retrieve the k-nearest training docs per query. Cheap with a TF-IDF index.",
    "**Prompt engineering ablation** — compare instruction phrasings, system-prompt vs user-prompt placement, and chain-of-thought reasoning prompts.",
    "**Calibration / abstention** — add a 'don't know' option to the few-shot prompt and measure if Llama uses it correctly on ambiguous cross-topic docs.",
]

REPRODUCIBILITY = """
- **Seed**: `SEED = 42` everywhere (`nlp_project.set_seed()` seeds Python `random`, NumPy, `PYTHONHASHSEED`, PyTorch). Word2vec uses an explicit `seed=` and `workers=1` because gensim multi-threading is non-deterministic.
- **Python**: 3.13, pinned in `.python-version`. Lock file `uv.lock` committed.
- **Tests**: 60+ pytest tests; `uv run pytest`, ~7s without `-m slow`.
- **Tracking**: every training run is logged to a single W&B project, grouped per question (`q1`, `q2`, `q3`, `q4`, `qbonus`).
- **AI tool usage**: Claude Code was used for scaffolding, test generation, and dashboard implementation. Disclosed in the report's "Tools used" section per spec.
"""
