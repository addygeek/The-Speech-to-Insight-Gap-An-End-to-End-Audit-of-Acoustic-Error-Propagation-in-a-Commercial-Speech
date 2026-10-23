"""Generate LaTeX tables and figures from measured results only."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging

logger = setup_logging()


def _stub_table(name: str, reason: str) -> str:
    return (
        f"% {name}: {reason}\n"
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{name} (data not available)}}\n"
        f"\\label{{tab:{name}}}\n\\end{{table}}\n"
    )


def write_tables(cfg: ExperimentConfig) -> None:
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.results_dir / "metrics.csv"

    if not metrics_path.is_file():
        for i, reason in enumerate(["missing metrics.csv"] * 3, start=1):
            (cfg.tables_dir / f"table{i}_{'dataset' if i==1 else 'wer' if i==2 else 'disagreement'}.tex").write_text(
                _stub_table(f"table{i}", reason), encoding="utf-8"
            )
        return

    df = pd.read_csv(metrics_path)
    n_utt = df["utterance_id"].nunique()
    conds = ", ".join(sorted(df["acoustic_condition"].unique()))
    table1 = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Dataset and acoustic conditions (measured).}\n\\label{tab:dataset}\n"
        "\\begin{tabular}{lr}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
        f"Utterances & {n_utt} \\\\\n"
        f"Metric rows & {len(df)} \\\\\n"
        f"Conditions & {conds} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    wer_by = df.groupby("acoustic_condition")["wer"].agg(["mean", "std", "count"]).reset_index()
    lines = ["\\begin{table}[t]\n\\centering\n\\caption{WER by acoustic condition.}\n\\label{tab:wer}\n"]
    lines.append("\\begin{tabular}{lrrr}\n\\toprule\nCondition & Mean WER & Std & N \\\\\n\\midrule\n")
    for _, row in wer_by.iterrows():
        lines.append(
            f"{row['acoustic_condition']} & {row['mean']:.3f} & {row['std']:.3f} & {int(row['count'])} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    table2 = "".join(lines)
