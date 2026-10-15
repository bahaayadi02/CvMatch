"""Evaluation experiment — the experimental core of this project.

We have 15 labelled (resume, JD, label) pairs, where label = 1 means
"good fit" and label = 0 means "wrong domain". We score each pair with
three different methods and compare:

    Method 1: TF-IDF cosine similarity (a strong, simple baseline).
    Method 2: Sentence-Transformer cosine similarity (semantic embedding).
    Method 3: Composite (semantic + tfidf + skill overlap, weighted).

Metrics:
    - ROC-AUC over the labelled pairs (does the score rank fits above non-fits?)
    - Mean score on positive pairs vs. negative pairs (separation)
    - Precision/Recall/F1 at a 0.5 threshold

Output:
    results/evaluation.csv   — per-pair scores from all three methods
    results/evaluation.md    — markdown summary table to paste into the report
    results/evaluation.png   — bar chart comparing methods
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matcher import (
    DEFAULT_WEIGHTS,
    semantic_similarity,
    skill_overlap_score,
    tfidf_similarity,
)
from src.parser import read_resume, read_job_description
from src.skill_extractor import extract_skills


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = PROJECT_ROOT / "data" / "labeled" / "pairs.csv"
RESUME_DIR = PROJECT_ROOT / "data" / "resumes"
JD_DIR = PROJECT_ROOT / "data" / "job_descriptions"
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_text(directory, file_id):
    """Find a file by its id (with .txt or .pdf extension)."""
    for ext in (".txt", ".pdf", ".md"):
        path = directory / f"{file_id}{ext}"
        if path.exists():
            if directory.name == "resumes":
                return read_resume(path)
            return read_job_description(path)
    raise FileNotFoundError(f"No file for id '{file_id}' in {directory}")


def main():
    pairs = pd.read_csv(PAIRS_PATH)
    print(f"Loaded {len(pairs)} labelled pairs from {PAIRS_PATH.name}")
    print(f"  Positive (good fit): {(pairs['label'] == 1).sum()}")
    print(f"  Negative (wrong fit): {(pairs['label'] == 0).sum()}\n")

    rows = []
    for _, row in pairs.iterrows():
        resume_text = _load_text(RESUME_DIR, row["resume_id"])
        jd_text = _load_text(JD_DIR, row["jd_id"])

        sem = semantic_similarity(resume_text, jd_text)
        tfidf = tfidf_similarity(resume_text, jd_text)
        skill = skill_overlap_score(
            extract_skills(resume_text), extract_skills(jd_text)
        )
        composite = (
            DEFAULT_WEIGHTS["semantic"] * sem
            + DEFAULT_WEIGHTS["tfidf"] * tfidf
            + DEFAULT_WEIGHTS["skill"] * skill
        )

        rows.append({
            "resume": row["resume_id"],
            "jd": row["jd_id"],
            "label": row["label"],
            "tfidf": tfidf,
            "semantic": sem,
            "skill_overlap": skill,
            "composite": composite,
            "note": row["note"],
        })
        print(
            f"  {row['resume_id']:<22} × {row['jd_id']:<22} "
            f"label={row['label']}  "
            f"tfidf={tfidf:.3f}  sem={sem:.3f}  composite={composite:.3f}  "
            f"({row['note']})"
        )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "evaluation.csv", index=False)

    # --- Per-method metrics ------------------------------------------------
    methods = ["tfidf", "semantic", "composite"]
    summary = []
    for m in methods:
        scores = df[m].to_numpy()
        labels = df["label"].to_numpy()
        # Pick a threshold: median works fairly across methods on this small set
        threshold = float(np.median(scores))
        preds = (scores >= threshold).astype(int)

        summary.append({
            "method": m,
            "auc": roc_auc_score(labels, scores),
            "mean_pos": scores[labels == 1].mean(),
            "mean_neg": scores[labels == 0].mean(),
            "separation": scores[labels == 1].mean() - scores[labels == 0].mean(),
            "f1": f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "threshold_used": threshold,
        })
    summary_df = pd.DataFrame(summary)

    print("\n=== Method comparison ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # --- Markdown table for the README/report ------------------------------
    md = ["# Evaluation results", "",
          f"Dataset: {len(df)} resume × JD pairs ({(df.label==1).sum()} positive, "
          f"{(df.label==0).sum()} negative).", "",
          "| Method | AUC | Mean (pos) | Mean (neg) | Separation | F1 |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        md.append(
            f"| {r['method']} | {r['auc']:.3f} | {r['mean_pos']:.3f} | "
            f"{r['mean_neg']:.3f} | {r['separation']:.3f} | {r['f1']:.3f} |"
        )
    (RESULTS_DIR / "evaluation.md").write_text("\n".join(md) + "\n")

    # --- Bar chart comparison ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.bar(summary_df["method"], summary_df["auc"], color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC (higher is better)")
    ax.set_title("Ranking quality on labelled pairs")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    width = 0.35
    x = np.arange(len(methods))
    ax.bar(x - width / 2, summary_df["mean_pos"], width,
           label="Mean score on POSITIVE pairs", color="seagreen")
    ax.bar(x + width / 2, summary_df["mean_neg"], width,
           label="Mean score on NEGATIVE pairs", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score")
    ax.set_title("Score separation between fits and non-fits")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "evaluation.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"\nWrote {RESULTS_DIR}/evaluation.csv, evaluation.md, evaluation.png")


if __name__ == "__main__":
    main()
