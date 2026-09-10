"""
Automated (non-LLM-judge) metrics. These are cheap, deterministic, and
reproducible -- run first, before spending any LLM-judge budget.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_report": classification_report(y_true, y_pred, zero_division=0),
    }


def escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """y_true / y_pred are "auto_handle" / "escalate" strings.
    Precision/recall computed with "escalate" as the positive class, since
    the costly failure mode is under-escalating (auto-handling something
    that needed a human), not over-escalating."""
    pos_label = "escalate"
    return {
        "escalation_precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "escalation_recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "escalation_f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "auto_handle_rate_pred": (pd.Series(y_pred) == "auto_handle").mean(),
        "auto_handle_rate_true": (pd.Series(y_true) == "auto_handle").mean(),
    }


def judge_human_agreement(judge_scores: list[float], human_scores: list[float]) -> dict:
    """Correlation between LLM-judge scores and human scores on the same
    subset. Required by the assignment ("evidence of how well your judge
    agrees with a human"). We report both Pearson (linear agreement) and
    Spearman (rank agreement, more robust to the judge/human using the
    1-5 scale slightly differently) plus exact & within-1 agreement rate,
    which is more interpretable than either correlation alone at small n."""
    s1 = pd.Series(judge_scores)
    s2 = pd.Series(human_scores)
    exact = (s1 == s2).mean()
    within_1 = (abs(s1 - s2) <= 1).mean()
    return {
        "pearson_r": s1.corr(s2, method="pearson"),
        "spearman_r": s1.corr(s2, method="spearman"),
        "exact_agreement_rate": exact,
        "within_1_point_agreement_rate": within_1,
        "n": len(s1),
    }
