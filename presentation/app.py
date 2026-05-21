"""HSLU NALAPRO — Presentation Dashboard.

Interactive single-page Streamlit app that drives the 20-minute final
presentation. Sidebar navigates 12 sections; every page is self-contained so
the presenter can jump to any experiment in any order during Q&A.

Run:    uv run streamlit run presentation/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

# Streamlit puts the script's directory on sys.path; we add the repo root too
# so `from presentation.x import ...` works whether the user launches the app
# from the repo root or from inside presentation/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import content  # noqa: E402
from data_loader import (  # noqa: E402
    CLASS_NAMES,
    cross_experiment_table,
    figure_path,
    load_q1_metrics,
    load_q4_metrics,
    load_qbonus_metrics,
    load_qbonus_sweep,
)

# ---------------------------------------------------------------------------
# Page config — runs once at import time

st.set_page_config(
    page_title="HSLU NALAPRO · Presentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light custom styling — minimal so the dashboard works whether Streamlit's
# default theme is light or dark.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1, h2 {margin-top: 0.5rem;}
      .breadcrumb {color: #888; font-size: 0.85rem; margin-bottom: 0.5rem;}
      .stMetric {background: rgba(127,127,127,0.06); padding: 0.5rem; border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Shared helpers


def breadcrumb(section: str) -> None:
    st.markdown(
        f"<div class='breadcrumb'>HSLU NALAPRO · {section}</div>",
        unsafe_allow_html=True,
    )


def rationale_block(question_key: str) -> None:
    """Render every DECISIONS[question_key] entry as a collapsed expander."""
    st.markdown("### 🧠 Methodology decisions — *why*")
    for entry in content.DECISIONS[question_key].values():
        with st.expander(f"**{entry['title']}**"):
            st.markdown(f"**What** — {entry['what']}")
            st.markdown(f"**Why** — {entry['why']}")
            st.caption(f"Source: `{entry['source']}`")


def render_confusion_matrix(cm: list[list[int]], title: str) -> go.Figure:
    """Interactive Plotly heatmap for a 20×20 confusion matrix."""
    cm_arr = np.array(cm)
    short = [c.split(".")[-1][:12] for c in CLASS_NAMES]
    fig = px.imshow(
        cm_arr,
        x=short,
        y=short,
        labels=dict(x="predicted", y="true", color="count"),
        color_continuous_scale="Blues",
        aspect="equal",
    )
    fig.update_layout(
        title=title,
        title_x=0.5,
        height=600,
        xaxis=dict(tickangle=-45),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_per_class_f1(
    per_class_f1: list[float], title: str, color: str = "#1f77b4"
) -> go.Figure:
    """Horizontal bar chart of per-class F1 (interactive — hover for value)."""
    df = pd.DataFrame(
        {"class": CLASS_NAMES, "f1": per_class_f1}
    ).sort_values("f1", ascending=True)
    fig = px.bar(
        df,
        x="f1",
        y="class",
        orientation="h",
        title=title,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        height=520,
        title_x=0.5,
        xaxis=dict(range=[0, 1], title="per-class F1"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1 — Overview


def page_overview() -> None:
    breadcrumb("1 / 12 · Overview")
    st.title(content.PROJECT_TITLE)
    st.markdown(f"#### {content.SUBTITLE}")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiments", "5", help="Q1 (3 reps + 1 ablation), Q2, Q3, Q4, Q-bonus")
    col2.metric("Models compared", "8", help="MLP×3, BERT, BERT+MLM, Llama×3, QLoRA")
    col3.metric("Best accuracy", "80.5%", help="Q-bonus QLoRA on Q4 200-doc subset")
    col4.metric("Time budget", "20 min", help="presentation + Q&A")

    st.markdown("### What this project is")
    for bullet in content.OVERVIEW_BULLETS:
        st.markdown(f"- {bullet}")

    st.markdown("### Talk outline (20 min)")
    outline = pd.DataFrame(
        [
            ["1–2", "Problem & dataset", "Why 20NG; class structure; preprocessing"],
            ["3–6", "Q1 — Shallow baselines", "word2vec / TF-IDF / mean+max → MLP"],
            ["7–9", "Q2 — BERT fine-tune", "HF Trainer, frozen vs full-fine-tune"],
            ["10–12", "Q3 — MLM domain-adapt", "Pretrain on 20NG, then fine-tune"],
            ["13–15", "Q4 — Llama zero/few-shot", "Frozen Llama-3, prompt-based"],
            ["16–18", "Q-bonus — QLoRA", "Parameter-efficient fine-tune of Llama"],
            ["19–20", "Cross-experiment + future work", "Headline comparison + Q&A teaser"],
        ],
        columns=["Min", "Section", "Content"],
    )
    st.dataframe(outline, hide_index=True, width="stretch")

    st.info(
        "**Navigation tip:** the sidebar on the left jumps to any section. "
        "Each experiment page has a *🧠 Methodology decisions* panel with "
        "expanders for every design choice — useful when answering Q&A."
    )


# ---------------------------------------------------------------------------
# Page 2 — Problem statement


def page_problem() -> None:
    breadcrumb("2 / 12 · Problem statement")
    st.title("The Problem")
    st.markdown(content.PROBLEM_STATEMENT)

    st.markdown("### Why this is harder than it looks")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Near-duplicate class pairs** make the upper accuracy bound "
            "tough. The model can't just match keywords — it has to "
            "discriminate between:"
        )
        st.code(
            "comp.sys.ibm.pc.hardware  ↔  comp.sys.mac.hardware\n"
            "talk.religion.misc        ↔  alt.atheism\n"
            "talk.politics.guns        ↔  talk.politics.misc\n"
            "rec.sport.baseball        ↔  rec.sport.hockey",
            language="text",
        )
    with col2:
        st.markdown(
            "**Label leakage if you're not careful.** Newsgroup posts "
            "come with headers like `Newsgroups: rec.autos` that spell "
            "the answer out. The spec mandates stripping these — see the "
            "first decision on the Q1 page."
        )


# ---------------------------------------------------------------------------
# Page 3 — Dataset


def page_dataset() -> None:
    breadcrumb("3 / 12 · Dataset")
    st.title("Dataset — 20 Newsgroups")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total documents", "18,846")
    col2.metric("Train / test", "11,314 / 7,532")
    col3.metric("Classes", "20")

    tab1, tab2 = st.tabs(["Class balance", "Document length"])
    with tab1:
        st.image(
            figure_path("class_balance.png"),
            caption="20NG classes are *nearly* balanced — ~600 docs/class in train, "
            "with a slight under-representation of `talk.religion.misc` (~376) "
            "and `alt.atheism` (~480). This is why we use macro-F1 for "
            "best-model selection.",
            width="stretch",
        )
    with tab2:
        st.image(
            figure_path("doc_length_hist.png"),
            caption="Document-length distribution (post-preprocessing). Heavy "
            "right tail — a few very long posts skew the mean, which is "
            "why BERT uses max_length=256 (Q2a histogram covers the 95th percentile).",
            width="stretch",
        )

    st.markdown("---")
    with st.expander("🧠 **Why `remove=('headers','footers','quotes')` is mandatory**"):
        d = content.DECISIONS["q1"]["remove_headers"]
        st.markdown(f"**What** — {d['what']}")
        st.markdown(f"**Why** — {d['why']}")
        st.caption(f"Source: `{d['source']}`")


# ---------------------------------------------------------------------------
# Page 4 — Method overview


def page_methods() -> None:
    breadcrumb("4 / 12 · Method overview")
    st.title("Five Experiments at a Glance")

    df = pd.DataFrame(
        [
            ["Q1b", "word2vec mean → MLP", "100-D dense", "trained from scratch", "Self-trained"],
            ["Q1c", "TF-IDF → MLP", "20k-D sparse", "—", "Sparse BoW"],
            ["Q1d", "word2vec mean+max → MLP", "200-D dense", "trained from scratch", "Self-trained"],
            ["Q2", "BERT fine-tune", "BERT [CLS]", "fine-tune full encoder", "bert-base-uncased"],
            ["Q3", "MLM-pretrain → fine-tune", "BERT [CLS]", "domain-adapt + fine-tune", "bert-base-uncased"],
            ["Q4", "Llama zero/few-shot", "prompt → generated label", "frozen (4-bit nf4)", "Llama-3.2-3B-Instruct"],
            ["Q-bonus", "QLoRA fine-tune", "score head", "LoRA on q/k/v/o + full head", "Llama-3.2-3B-Instruct"],
        ],
        columns=["#", "Method", "Representation", "Training", "Base model"],
    )
    st.dataframe(df, hide_index=True, width="stretch")

    st.markdown("### Progression of paradigms")
    st.markdown(
        "The five experiments deliberately trace the **last decade of NLP**:\n\n"
        "1. **Bag-of-words & shallow embeddings** (Q1) — pre-2017, the workhorse era of TF-IDF and word2vec.\n"
        "2. **Pre-trained transformers** (Q2) — 2018+, BERT and friends.\n"
        "3. **Domain-adaptive pretraining** (Q3) — 2020 (Gururangan et al.), 'don't stop pretraining'.\n"
        "4. **Prompt-based LLMs** (Q4) — 2022+, GPT-3 → Llama, zero/few-shot.\n"
        "5. **Parameter-efficient fine-tuning** (Q-bonus) — 2023+, QLoRA on quantised base models.\n\n"
        "Every step buys something — but as we'll see, **simpler isn't always worse on 20NG**."
    )


# ---------------------------------------------------------------------------
# Page 5 — Q1


def page_q1() -> None:
    breadcrumb("5 / 12 · Q1 — Shallow baselines")
    st.title("Q1 — MLP with three input representations")
    st.caption("Two-layer MLP (Linear → ReLU → Linear) trained from scratch on three input reps.")

    metrics = load_q1_metrics()

    # Headline metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Q1b · word2vec mean", f"{metrics.iloc[0]['accuracy']:.1%}",
                f"macro-F1 {metrics.iloc[0]['macro_f1']:.3f}")
    col2.metric("Q1c · TF-IDF", f"{metrics.iloc[1]['accuracy']:.1%}",
                f"macro-F1 {metrics.iloc[1]['macro_f1']:.3f}")
    col3.metric("Q1d · w2v mean+max", f"{metrics.iloc[2]['accuracy']:.1%}",
                f"macro-F1 {metrics.iloc[2]['macro_f1']:.3f}")

    st.markdown("### Pick a variant to inspect")
    variant = st.selectbox(
        "Q1 variant",
        options=["Q1b — word2vec mean-pool", "Q1c — TF-IDF", "Q1d — word2vec mean+max-pool"],
        index=1,
        label_visibility="collapsed",
    )

    cm_map = {
        "Q1b — word2vec mean-pool": "confusion_matrix_q1b.png",
        "Q1c — TF-IDF": "confusion_matrix_q1c.png",
        "Q1d — word2vec mean+max-pool": "confusion_matrix_q1d.png",
    }
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.image(figure_path(cm_map[variant]), caption=f"Confusion matrix — {variant}", width="stretch")
    with col_b:
        # Comparison bar chart — accuracy + macro-F1 side by side
        m_long = metrics.melt(
            id_vars=["experiment"], value_vars=["accuracy", "macro_f1"],
            var_name="metric", value_name="value",
        )
        fig = px.bar(
            m_long, x="experiment", y="value", color="metric",
            barmode="group", title="Q1 — headline comparison",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"],
        )
        fig.update_layout(
            height=420, title_x=0.5,
            yaxis=dict(range=[0, 1]), xaxis=dict(title=""),
            margin=dict(l=10, r=10, t=50, b=80),
        )
        fig.update_xaxes(tickangle=-15)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Word2vec embedding evolution (Q1b)")
    st.image(
        figure_path("w2v_tsne_epoch1_vs_epoch20.png"),
        caption="t-SNE of word2vec embeddings — 1 epoch (left) vs 20 epochs (right). "
        "Domain-specific clusters (`hockey`, `nasa`, `windows`) only emerge after "
        "multiple passes through the corpus. The 1-epoch snapshot is still close to "
        "the random initialisation neighborhood.",
        width="stretch",
    )

    st.markdown("---")
    rationale_block("q1")


# ---------------------------------------------------------------------------
# Page 6 — Q2


def page_q2() -> None:
    breadcrumb("6 / 12 · Q2 — BERT fine-tune")
    st.title("Q2 — Fine-tuning `bert-base-uncased`")
    st.caption("HuggingFace Trainer, MPS-safe flags, two-stage Q1↔Q2 comparison.")

    tabs = st.tabs(["Baseline (Q2b)", "Hyperparameter sweep (Q2c)", "Q1 vs Q2 comparison (Q2d)", "Token-length analysis (Q2a)"])

    with tabs[0]:
        st.image(
            figure_path("q2b_confusion_matrix.png"),
            caption="Q2b confusion matrix — BERT fine-tuned with lr=2e-5, seq=256, "
            "3 epochs. Dramatic improvement over Q1: off-diagonal mass collapses; "
            "remaining errors concentrate on the near-duplicate class pairs "
            "(`comp.sys.*`, `talk.politics.*`).",
            width="stretch",
        )
    with tabs[1]:
        st.image(
            figure_path("q2c_sweep_bars.png"),
            caption="Q2c sweep over learning rate × epochs × max_length. Best "
            "configuration written to `models/q2_results/q2c_sweep.json` and "
            "promoted as Q2b baseline.",
            width="stretch",
        )
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.image(figure_path("q2d_comparison_bars.png"),
                     caption="Accuracy + macro-F1 — all Q1 variants vs Q2b.",
                     width="stretch")
        with col2:
            st.image(figure_path("q2d_q2b_cm.png"),
                     caption="Same Q2b CM, included for direct side-by-side reading.",
                     width="stretch")
    with tabs[3]:
        st.image(
            figure_path("q2a_token_length_hist.png"),
            caption="BERT-tokenized document length distribution. The 95th "
            "percentile sits ≈250 tokens; we set max_length=256 for the optimal "
            "compute/coverage trade-off.",
            width="stretch",
        )

    st.markdown("---")
    rationale_block("q2")


# ---------------------------------------------------------------------------
# Page 7 — Q3


def page_q3() -> None:
    breadcrumb("7 / 12 · Q3 — MLM → fine-tune")
    st.title("Q3 — Domain-adaptive MLM then classification fine-tune")
    st.caption(
        "Stage A: MLM pretrain on 20NG train texts. Stage B: classification "
        "fine-tune with Q2b's hyperparameters held constant."
    )

    tabs = st.tabs(["Stage A — MLM loss", "Stage B — confusion matrix", "Q2 vs Q3 comparison"])

    with tabs[0]:
        st.image(
            figure_path("q3_mlm_loss.png"),
            caption="Stage A MLM training loss. Internal 90/10 eval split "
            "tracks perplexity without leaking the classification val set.",
            width="stretch",
        )
    with tabs[1]:
        st.image(
            figure_path("q3_confusion_matrix.png"),
            caption="Stage B confusion matrix — same BERT recipe as Q2b, "
            "but starting from the MLM-adapted checkpoint.",
            width="stretch",
        )
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.image(figure_path("q3_vs_q2_bars.png"),
                     caption="Headline metrics — Q2 baseline vs Q3 MLM-then-FT.",
                     width="stretch")
        with col2:
            st.image(figure_path("q3_vs_q2_confusion_side_by_side.png"),
                     caption="Side-by-side CMs — does MLM help the hard class pairs?",
                     width="stretch")

    st.markdown("---")
    rationale_block("q3")


# ---------------------------------------------------------------------------
# Page 8 — Q4


def page_q4() -> None:
    breadcrumb("8 / 12 · Q4 — Llama zero/few-shot")
    st.title("Q4 — Frozen Llama-3.2-3B-Instruct, prompt-based classification")
    st.caption("4-bit nf4 quantisation, greedy decoding, 200-doc stratified subsample.")

    q4 = load_q4_metrics()

    cond_map = {"zero": "Zero-shot", "1pc": "Few-shot (k=1/class)", "3pc": "Few-shot (k=3/class)"}
    rev_map = {v: k for k, v in cond_map.items()}

    # Three-up headline metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Zero-shot", f"{q4['zero']['accuracy']:.1%}", f"macro-F1 {q4['zero']['macro_f1']:.3f}")
    col2.metric("k=1/class", f"{q4['1pc']['accuracy']:.1%}", f"macro-F1 {q4['1pc']['macro_f1']:.3f}")
    col3.metric("k=3/class", f"{q4['3pc']['accuracy']:.1%}", f"macro-F1 {q4['3pc']['macro_f1']:.3f}")

    st.markdown("### Drill into one condition")
    label = st.selectbox(
        "Condition", list(cond_map.values()), index=2, label_visibility="collapsed"
    )
    key = rev_map[label]
    d = q4[key]

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            render_confusion_matrix(d["confusion_matrix"], f"{label} — confusion matrix"),
            width="stretch",
        )
    with col_b:
        st.plotly_chart(
            render_per_class_f1(d["per_class_f1"], f"{label} — per-class F1"),
            width="stretch",
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{d['accuracy']:.1%}")
    col2.metric("Macro-F1", f"{d['macro_f1']:.3f}")
    inv = d.get("invalid_rate")
    col3.metric(
        "Invalid-output rate",
        f"{inv:.1%}" if inv is not None else "n/a",
        help="Share of Llama outputs that needed substring or Levenshtein-fallback parsing",
    )

    st.markdown("### Comparison across all three conditions")
    st.image(
        figure_path("q4_comparison_bars.png"),
        caption="Few-shot demos give Llama a clear lift — but it's still ~30pp "
        "below a fine-tuned BERT and ~45pp below QLoRA on the same subset.",
        width="stretch",
    )

    st.markdown("---")
    rationale_block("q4")


# ---------------------------------------------------------------------------
# Page 9 — Q-bonus


def page_qbonus() -> None:
    breadcrumb("9 / 12 · Q-bonus — QLoRA")
    st.title("Q-bonus — QLoRA fine-tune of Llama-3.2-3B-Instruct")
    st.caption(
        "Two-run sweep (r=16, lr=2e-4) vs (r=32, lr=1e-4); winner evaluated on "
        "both the full test set and the Q4 200-doc subset."
    )

    qb = load_qbonus_metrics()
    sweep = load_qbonus_sweep()

    # Headline metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Full 7,532-doc test", f"{qb['full_test']['accuracy']:.1%}",
        f"macro-F1 {qb['full_test']['macro_f1']:.3f}",
    )
    col2.metric(
        "Q4 200-doc subset", f"{qb['q4subset']['accuracy']:.1%}",
        f"macro-F1 {qb['q4subset']['macro_f1']:.3f}",
    )
    col3.metric(
        "Winning sweep run", sweep["winner"],
        help="Selected by val macro-F1",
    )

    tabs = st.tabs(["Sweep summary", "Full-test confusion matrix", "Per-class delta vs zero-shot", "Final comparison"])

    with tabs[0]:
        runs = sweep["runs"]
        sweep_df = pd.DataFrame(
            [
                {
                    "run_name": r["run_name"],
                    "lora_r": r["config"]["lora_r"],
                    "lora_alpha": r["config"]["lora_alpha"],
                    "learning_rate": r["config"]["learning_rate"],
                    "epochs": r["config"]["num_train_epochs"],
                    "val_accuracy": round(r["val_metrics"]["eval_accuracy"], 4),
                    "val_macro_f1": round(r["val_metrics"]["eval_macro_f1"], 4),
                    "val_loss": round(r["val_metrics"]["eval_loss"], 4),
                }
                for r in runs
            ]
        )
        st.dataframe(sweep_df, hide_index=True, width="stretch")
        st.caption(
            f"Winner: **{sweep['winner']}** — selected by val macro-F1. "
            "Higher rank (r=32) under-performed despite the lower LR, suggesting "
            "the extra adapter capacity was unhelpful on 20NG's signal at this "
            "training budget."
        )

    with tabs[1]:
        st.plotly_chart(
            render_confusion_matrix(
                qb["full_test"]["confusion_matrix"],
                "Q-bonus QLoRA — full 7,532-doc test set",
            ),
            width="stretch",
        )

    with tabs[2]:
        st.image(
            figure_path("qbonus_per_class_delta.png"),
            caption="Per-class F1 improvement from QLoRA fine-tune over Llama "
            "zero-shot — every class improves, with the biggest gains on the "
            "tech classes that zero-shot Llama struggled to disambiguate.",
            width="stretch",
        )

    with tabs[3]:
        st.image(
            figure_path("qbonus_comparison_bars.png"),
            caption="Headline comparison — QLoRA vs Q4 zero/few-shot on the "
            "byte-identical 200-doc subset. ~+60pp accuracy gain from fine-tuning "
            "a quantised base model with ≈0.6% of parameters trainable.",
            width="stretch",
        )

    st.markdown("---")
    rationale_block("qbonus")


# ---------------------------------------------------------------------------
# Page 10 — Cross-experiment comparison


def page_cross() -> None:
    breadcrumb("10 / 12 · Cross-experiment comparison")
    st.title("Putting it all together")
    st.caption(
        "Live, interactive comparison across every experiment that produced a "
        "JSON or CSV metric dump. (Q2/Q3 numerical baselines live in the per-page "
        "figures — embed-only on this branch by design.)"
    )

    df = cross_experiment_table()

    col1, col2 = st.columns(2)
    with col1:
        metric = st.radio(
            "Metric", ["accuracy", "macro_f1"], horizontal=True, index=0
        )
    with col2:
        sort_desc = st.toggle("Sort descending", value=True)

    df_sorted = df.sort_values(metric, ascending=not sort_desc)

    fig = px.bar(
        df_sorted,
        x=metric, y="experiment", orientation="h", color="source",
        title=f"All experiments — {metric}",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text=df_sorted[metric].apply(lambda x: f"{x:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=520, title_x=0.5,
        xaxis=dict(range=[0, 1]), yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Two stories in one chart")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Top of the chart** — QLoRA on the Q4 subset hits **80.5% acc / "
            "macro-F1 0.80**, beating every other approach. Q-bonus on the *full* "
            "test is 75.7% — the gap to Q2/Q3 BERT (~82% on the full test, see "
            "their figures) suggests BERT's denser parameter-tuning still wins "
            "when raw classification accuracy is the only objective."
        )
    with col2:
        st.markdown(
            "**Bottom of the chart** — frozen Llama zero-shot at **15.5%** "
            "(barely 3× random for 20 classes). Few-shot demos move the needle "
            "linearly to 36.5% by k=3, but the bag-of-words TF-IDF + MLP "
            "still beats few-shot Llama by 30+ points."
        )

    st.dataframe(df_sorted, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page 11 — Conclusions & future work


def page_conclusions() -> None:
    breadcrumb("11 / 12 · Conclusions & future work")
    st.title("Key findings")

    for f in content.KEY_FINDINGS:
        st.markdown(f"- {f}")

    st.divider()
    st.markdown("### Future work")
    for f in content.FUTURE_WORK:
        st.markdown(f"- {f}")

    st.divider()
    st.markdown("### What I would do differently")
    st.markdown(
        "- **Run Q4 on the full test set** with batched inference (vLLM or HF "
        "`pipeline`) to get a paired comparison against BERT and QLoRA on the same "
        "7,532 docs — the 200-doc subset is the only paired evidence at Llama scale.\n"
        "- **Add a third QLoRA sweep point** at (r=8, lr=3e-4) — the canonical "
        "low-rank baseline. Would let me triangulate whether the rank=16 sweet "
        "spot is real or a function of LR choice.\n"
        "- **Bootstrap-CI every headline metric** so the cross-experiment bar "
        "chart shows error bars, not point estimates."
    )


# ---------------------------------------------------------------------------
# Page 12 — Appendix


def page_appendix() -> None:
    breadcrumb("12 / 12 · Appendix")
    st.title("Appendix")

    st.markdown("### Weights & Biases")
    st.markdown(
        f"All training runs (Q1–Q-bonus, ~30 runs) logged to a single project: "
        f"[hslu-nalapro on W&B]({content.WANDB_URL})"
    )

    st.markdown("### Reproducibility")
    st.markdown(content.REPRODUCIBILITY)

    st.markdown("### Citations")
    st.markdown(
        """
- Mikolov et al. (2013) — *Efficient estimation of word representations in vector space.* (word2vec)
- Devlin et al. (2019) — *BERT: Pre-training of deep bidirectional transformers for language understanding.*
- Gururangan et al. (2020) — *Don't stop pretraining: adapt language models to domains and tasks.* (DAPT)
- Touvron et al. (2024) — *The Llama 3 Herd of Models.*
- Hu et al. (2021) — *LoRA: Low-rank adaptation of large language models.*
- Dettmers et al. (2023) — *QLoRA: efficient finetuning of quantized LLMs.*
- Pedregosa et al. (2011) — *Scikit-learn: machine learning in Python.* (`fetch_20newsgroups`, TF-IDF)
- Wolf et al. (2020) — *Transformers: state-of-the-art natural language processing.* (HuggingFace)
"""
    )

    st.markdown("### AI tool disclosure")
    st.markdown(
        "Per NALAPRO §4, Claude Code (Anthropic) was used for:\n"
        "- Scaffolding the `src/nlp_project/` package modules and pytest suite.\n"
        "- Generating notebook boilerplate via `scripts/build_q{2,3,4,bonus}_notebook.py`.\n"
        "- Implementing this presentation dashboard.\n\n"
        "All experimental design decisions, hyperparameter choices, and the report "
        "narrative are the author's own. Every AI-suggested code block was "
        "reviewed and tested before commit; non-trivial logic carries unit tests."
    )

    st.markdown("### Code")
    st.markdown(
        "- Repo layout: see `CLAUDE.md` in the project root.\n"
        "- Tests: `uv run pytest` — 60+ tests, ~7s without `-m slow`.\n"
        "- Dashboard source: `presentation/` (this app)."
    )


# ---------------------------------------------------------------------------
# Sidebar nav + dispatch

PAGES = {
    "1 · Overview": page_overview,
    "2 · Problem statement": page_problem,
    "3 · Dataset": page_dataset,
    "4 · Method overview": page_methods,
    "5 · Q1 — Shallow baselines": page_q1,
    "6 · Q2 — BERT fine-tune": page_q2,
    "7 · Q3 — MLM → fine-tune": page_q3,
    "8 · Q4 — Llama zero/few-shot": page_q4,
    "9 · Q-bonus — QLoRA": page_qbonus,
    "10 · Cross-experiment comparison": page_cross,
    "11 · Conclusions & future work": page_conclusions,
    "12 · Appendix": page_appendix,
}


def main() -> None:
    st.sidebar.title("📊 Presentation")
    st.sidebar.caption(content.SUBTITLE)
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("**Live talk · 20 min**")
    st.sidebar.caption(
        "Sidebar jumps you to any section in any order — useful during Q&A "
        "for surfacing a specific decision rationale or confusion matrix."
    )
    st.sidebar.markdown(f"[🔗 W&B project]({content.WANDB_URL})")

    PAGES[choice]()


if __name__ == "__main__":
    main()
