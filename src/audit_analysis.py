"""Analysis for the harsher conditions (E6) and the corrected statistics (E10).

Adds the manipulation check (did the perturbation move WER?), the dose-response of label disagreement on
per-item WER with a logistic model, and Holm-corrected paired tests against the clean condition.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.cache import ACOUSTIC_CONDITIONS, cache_hit, load_json, raw_response_path
from src.calculate_wer import compute_wer
from src.config_loader import ExperimentConfig
from src.extra_conditions import EXTRA_CONDITIONS, listen_extra_path, read_extra_path
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.parse_responses import extract_transcript
from src.second_voice import VOICE_CONDITIONS
from src.stability_baseline import _labels
from src.statistical_analysis import _bootstrap_mean

logger = setup_logging()

ALL_CONDITIONS = (*ACOUSTIC_CONDITIONS, *EXTRA_CONDITIONS, *VOICE_CONDITIONS)


def _transcript(cfg: ExperimentConfig, uid: str, cond: str) -> str:
    path = raw_response_path(cfg, "listen", f"{uid}_{cond}") if cond in ACOUSTIC_CONDITIONS else listen_extra_path(cfg, uid, cond)
    if not cache_hit(path):
        return ""
    payload = load_json(path)
    return extract_transcript(payload.get("response") or payload)


def _read_labels(cfg: ExperimentConfig, uid: str, cond: str):
    path = raw_response_path(cfg, "read", f"{uid}_transcript_{cond}") if cond in ACOUSTIC_CONDITIONS else read_extra_path(cfg, uid, cond)
    return _labels(cfg, path)


def item_frame(cfg: ExperimentConfig, split: str | None = "full") -> pd.DataFrame:
    """One row per item and condition: WER and label disagreement with the reference-text analysis."""
    rows: list[dict[str, Any]] = []
    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        ref = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_reference"))
        for cond in ALL_CONDITIONS:
            text = _transcript(cfg, uid, cond)
            if not text:
                continue
            row: dict[str, Any] = {"utterance_id": uid, "condition": cond, "wer": compute_wer(rec["reference_text"], text, cfg).wer}
            hyp = _read_labels(cfg, uid, cond)
            if ref and hyp:
                row["intent_disagree"] = float(ref["intent"] != hyp["intent"])
                row["topic_disagree"] = float(ref["topics"] != hyp["topics"])
                row["sentiment_disagree"] = float(ref["sentiment"] != hyp["sentiment"])
            rows.append(row)
    return pd.DataFrame(rows)


def dose_response(frame: pd.DataFrame) -> dict[str, Any]:
    """Logistic regression of each disagreement flag on per-item WER, pooled over conditions."""
    out: dict[str, Any] = {}
    for flag in ("intent_disagree", "topic_disagree", "sentiment_disagree"):
        data = frame.dropna(subset=[flag])
        y = data[flag].to_numpy()
        if len(data) < 30 or y.min() == y.max():
            out[flag] = {"n": len(data), "note": "no variation in outcome" if len(data) else "no data"}
            continue
        model = sm.Logit(y, sm.add_constant(data[["wer"]].to_numpy())).fit(disp=0)
        lo, hi = model.conf_int()[1]
        out[flag] = {"n": len(data), "odds_ratio_per_0.1_wer": float(np.exp(0.1 * model.params[1])), "or_lo": float(np.exp(0.1 * lo)), "or_hi": float(np.exp(0.1 * hi)), "p": float(model.pvalues[1])}
    bins = pd.cut(frame["wer"], [-0.001, 0.0, 0.05, 0.15, 0.3, 10], labels=["0", "(0,0.05]", "(0.05,0.15]", "(0.15,0.30]", ">0.30"])
    out["rate_by_wer_bin"] = {
        str(b): {"n": int(len(g)), "intent_disagree": float(g["intent_disagree"].mean()), "topic_disagree": float(g["topic_disagree"].mean())}
        for b, g in frame.assign(wer_bin=bins).groupby("wer_bin", observed=True)
        if len(g)
    }
    return out


def manipulation_check(frame: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    """Mean WER per condition and the paired difference to clean with a Wilcoxon p-value, Holm-corrected."""
    clean = frame[frame.condition == "clean"].set_index("utterance_id")["wer"]
    out: dict[str, Any] = {}
    pvals, names = [], []
    for cond in ALL_CONDITIONS:
        sub = frame[frame.condition == cond].set_index("utterance_id")["wer"]
        both = clean.index.intersection(sub.index)
        if len(both) < 10:
            continue
        diff = (sub.loc[both] - clean.loc[both]).to_numpy()
        mean, lo, hi = _bootstrap_mean(diff, n_boot, 42)
        p = 1.0 if cond == "clean" or np.allclose(diff, 0) else float(stats.wilcoxon(diff, zero_method="zsplit").pvalue)
        out[cond] = {"n": int(len(both)), "wer_mean": float(sub.loc[both].mean()), "paired_diff": mean, "diff_lo": lo, "diff_hi": hi, "p_raw": p, "items_changed": int((diff != 0).sum())}
        if cond != "clean":
            names.append(cond)
            pvals.append(p)
    if pvals:
        for name, adj in zip(names, multipletests(pvals, method="holm")[1]):
            out[name]["p_holm"] = float(adj)
    return out


def analyze_conditions(cfg: ExperimentConfig, split: str | None = "full") -> dict[str, Any]:
    n_boot = int(cfg.section("statistics").get("bootstrap_samples", 2000))
    frame = item_frame(cfg, split)
    frame.to_csv(cfg.results_dir / "item_conditions.csv", index=False)
    labelled = frame.dropna(subset=["intent_disagree"]) if "intent_disagree" in frame else frame
    per_condition = {
        cond: {
            "n": int(len(g)),
            "intent_disagree": float(g["intent_disagree"].mean()),
            "topic_disagree": float(g["topic_disagree"].mean()),
            "sentiment_disagree": float(g["sentiment_disagree"].mean()),
        }
        for cond, g in labelled.groupby("condition")
    }
    result = {"manipulation_check": manipulation_check(frame, n_boot), "label_disagreement_vs_reference": per_condition, "dose_response": dose_response(labelled)}
    path = cfg.results_dir / "conditions_extra.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", path)
    return result
