"""LaTeX tables and figures for the follow-up experiments, built from the saved result JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging

logger = setup_logging()

COND_LABEL = {
    "clean": "Clean", "mild": "15 dB pink", "strong": "8 dB pink", "harsh": "0 dB pink",
    "pinkm5": "$-5$ dB pink", "pinkm10": "$-10$ dB pink", "babble5": "5 dB babble", "babble0": "0 dB babble",
    "reverb": "Reverb 0.8 s", "phone": "Telephone codec", "phonebabble": "Phone + 5 dB babble",
    "orion_clean": "Orion, clean", "orion_harsh": "Orion, 0 dB", "luna_clean": "Luna, clean", "luna_harsh": "Luna, 0 dB",
}


def _load(cfg: ExperimentConfig, name: str) -> dict[str, Any]:
    path = cfg.results_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pct(block: dict[str, float]) -> str:
    return f"{100 * block['mean']:.1f} [{100 * block['lo']:.1f}, {100 * block['hi']:.1f}]"


def _table(path: Path, caption: str, label: str, header: str, spec: str, rows: list[str]) -> None:
    body = " \\\\\n".join(rows)
    path.write_text(
        "\\begin{table}[t]\n\\centering\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\resizebox{{\\linewidth}}{{!}}{{\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n",
        encoding="utf-8",
    )


def write_stability_table(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "stability.json")
    if not d:
        return
    rows = []
    for source, b in d["noise_floor"].items():
        name = "Same reference text, repeated calls" if source == "reference" else "Same clean transcript, repeated calls"
        rows.append(f"{name} & {b['items']} & {_pct(b['intent_disagree'])} & {_pct(b['topic_disagree'])} & {100 * b['sentiment_disagree']['mean']:.1f}")
    for cond, b in d["cross_condition"].items():
        label = COND_LABEL[cond.removeprefix("reference_vs_")]
        rows.append(f"Reference vs {label} transcript & {b['intent_disagree']['n']} & {_pct(b['intent_disagree'])} & {_pct(b['topic_disagree'])} & {100 * b['sentiment_disagree']['mean']:.1f}")
    b = d["clean_vs_harsh_identical_transcript"]
    rows.append(f"Clean vs 0 dB, identical transcript & {b['items']} & {_pct(b['intent_disagree'])} & {_pct(b['topic_disagree'])} & {100 * b['sentiment_disagree']['mean']:.1f}")
    for cond, b in d["exact_match_subset"].items():
        c = b["byte_identical"]
        rows.append(f"Reference vs {COND_LABEL[cond]}, byte-identical text only & {c['items']} & {_pct(c['intent_disagree'])} & {_pct(c['topic_disagree'])} & {100 * c['sentiment_disagree']['mean']:.1f}")
    _table(
        cfg.tables_dir / "table_stability.tex",
        f"Label disagreement (\\%, bootstrap 95\\% interval) on identical input and across conditions. Repeated calls use {d['repeats_per_item']} calls per item. The first two rows are the noise floor.",
        "tab:stability",
        "Comparison & Items & Intent & Topic set & Sentiment",
        "lrrrr",
        rows,
    )


def write_conditions_table(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "conditions_extra.json")
    if not d:
        return
    mc, lab = d["manipulation_check"], d["label_disagreement_vs_reference"]
    rows = []
    for cond, m in mc.items():
        l = lab.get(cond, {})
        diff = "--" if cond == "clean" else f"{m['paired_diff']:+.4f} [{m['diff_lo']:+.4f}, {m['diff_hi']:+.4f}]"
        holm = "--" if "p_holm" not in m else f"{m['p_holm']:.1e}"
        rows.append(
            f"{COND_LABEL[cond]} & {m['n']} & {m['wer_mean']:.3f} & {diff} & {holm} & "
            f"{100 * l.get('intent_disagree', float('nan')):.1f} & {100 * l.get('topic_disagree', float('nan')):.1f} & {100 * l.get('sentiment_disagree', float('nan')):.1f}"
        )
    _table(
        cfg.tables_dir / "table_conditions.tex",
        "Manipulation check and label disagreement with the reference-text analysis (\\%). Paired WER difference is to the clean Asteria condition (bootstrap 95\\% interval; Holm-adjusted Wilcoxon $p$).",
        "tab:conditions",
        "Condition & $n$ & WER & Paired $\\Delta$WER & $p_{\\mathrm{Holm}}$ & Intent & Topic & Sent.",
        "lrrrrrrr",
        rows,
    )


def write_semantic_table(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "semantic_match.json")
    if not d:
        return
    names = {
        "noise_floor_reference": "Reference text, repeated call",
        "reference_vs_clean": "Reference vs clean transcript",
        "clean_vs_harsh_same_transcript": "Clean vs 0 dB, identical transcript",
    }
    rows = []
    for key, g in d["groups"].items():
        i, t = g["intent"], g["topic"]
        rows.append(
            f"{names[key]} & {g['items']} & {100 * i['string_disagreement']:.1f} & {100 * i['semantic_disagreement']['mean']:.1f} & {100 * i['paraphrase_fraction']['mean']:.1f} & "
            f"{100 * t['string_disagreement']:.1f} & {100 * t['semantic_disagreement']['mean']:.1f} & {100 * t['paraphrase_fraction']['mean']:.1f}"
        )
    _table(
        cfg.tables_dir / "table_semantic.tex",
        f"String versus semantic disagreement (\\%). Semantic = string-different and judged a different goal or subject by {d['judge_model'].split('/')[-1]} (temperature 0). Para. = share of string disagreements judged equivalent.",
        "tab:semantic",
        "Comparison & Items & \\multicolumn{3}{c}{Intent: str. / sem. / para.} & \\multicolumn{3}{c}{Topic: str. / sem. / para.}",
        "lrrrrrrr",
        rows,
    )


def write_actions_table(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "actions.json")
    if not d:
        return
    rows = []

    def row(name: str, b: dict[str, Any]) -> str:
        return f"{name} & {b['action']['n']} & {_pct(b['route'])} & {_pct(b['escalate'])} & {_pct(b['action'])}"

    rows.append(row("Same reference text, repeated call (floor)", d["same_text_repeat_floor"]))
    for cond, b in d["vs_reference"].items():
        if b["action"]["n"]:
            rows.append(row(f"Reference vs {COND_LABEL[cond]}", b))
    _table(
        cfg.tables_dir / "table_actions.tex",
        "Downstream action disagreement (\\%). Route is the queue the intent phrase maps to; escalate is negative sentiment; action is either differing. The first row is the instability floor.",
        "tab:actions",
        "Comparison & Items & Route & Escalate & Action",
        "lrrrr",
        rows,
    )


def write_llm_table(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "llm_stability.json")
    if not d:
        return
    rows = []
    for model, temps in d["models"].items():
        for temp, b in temps.items():
            if b["items_usable"]:
                rows.append(
                    f"{model.split('/')[-1]} & {float(temp):.0f} & {b['items_usable']} & {_pct(b['intent_string_disagreement'])} & {_pct(b['intent_semantic_disagreement'])} & {_pct(b['sentiment_disagreement'])}"
                )
    _table(
        cfg.tables_dir / "table_llm.tex",
        f"Repeat disagreement of an LLM labeler on identical reference text ({d['repeats']} calls per item), by temperature (\\%).",
        "tab:llm",
        "Model & Temp. & Items & Intent (string) & Intent (semantic) & Sentiment",
        "lrrrrr",
        rows,
    )


def plot_floor(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "stability.json")
    if not d:
        return
    labels = ["Repeat\n(ref.)", "Repeat\n(transcript)", "Clean", "15 dB", "8 dB", "0 dB"]
    blocks = [d["noise_floor"]["reference"], d["noise_floor"]["transcript_clean"]] + [d["cross_condition"][f"reference_vs_{c}"] for c in ("clean", "mild", "strong", "harsh")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, key, title in zip(axes, ("intent_disagree", "topic_disagree"), ("Intent string", "Topic set")):
        means = [100 * b[key]["mean"] for b in blocks]
        err = [[100 * (b[key]["mean"] - b[key]["lo"]) for b in blocks], [100 * (b[key]["hi"] - b[key]["mean"]) for b in blocks]]
        ax.bar(range(len(blocks)), means, yerr=err, color=["#7F8C8D"] * 2 + ["#1B4F72"] * 4, capsize=2)
        ax.set_xticks(range(len(blocks)), labels, fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Disagreement (%)")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "fig13_noise_floor.pdf")
    plt.close(fig)


def plot_dose(cfg: ExperimentConfig) -> None:
    d = _load(cfg, "conditions_extra.json")
    s = _load(cfg, "stability.json")
    if not d or not s:
        return
    mc, lab = d["manipulation_check"], d["label_disagreement_vs_reference"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for cond, m in mc.items():
        if cond in lab:
            ax.scatter(100 * m["wer_mean"], 100 * lab[cond]["intent_disagree"], s=22, color="#1B4F72")
    floor = s["noise_floor"]["reference"]["intent_disagree"]
    ax.axhspan(100 * floor["lo"], 100 * floor["hi"], color="#7F8C8D", alpha=0.3, label="Same-text repeat floor")
    ax.set_xlabel("Mean word error (%)")
    ax.set_ylabel("Intent disagreement (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "fig14_dose_response.pdf")
    plt.close(fig)


def run_followup_assets(cfg: ExperimentConfig) -> None:
    for step in (write_stability_table, write_conditions_table, write_semantic_table, write_actions_table, write_llm_table, plot_floor, plot_dose):
        step(cfg)
    logger.info("Wrote follow-up tables and figures")
