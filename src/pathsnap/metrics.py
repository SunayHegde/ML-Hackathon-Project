"""Evaluation metrics for candidate classification and route reconstruction."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        cand_accuracy=accuracy_score(y, yhat),
        macro_f1=f1_score(y, yhat, average="macro"),
        macro_precision=precision_score(y, yhat, average="macro", zero_division=0),
        macro_recall=recall_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
    )


def top1_predictions(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Arg-max candidate per (trip, obs)."""
    d = df[["trip", "obs", "arc", "y"]].copy()
    d["p"] = p
    idx = d.groupby(["trip", "obs"])["p"].idxmax()
    return d.loc[idx].reset_index(drop=True)


def top1_accuracy(top1: pd.DataFrame, n_obs_total: int) -> float:
    """Observation-level Top-1 accuracy. Fixes with no candidates count as wrong."""
    return float(top1["y"].sum() / n_obs_total)


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive predicted (non-missing) arc pairs that are the
    same arc or directly connected (to_node(a) == from_node(b))."""
    a = np.asarray([x for x in pred_arcs if x is not None and x >= 0], dtype=np.int64)
    if len(a) < 2:
        return float("nan")
    same = a[1:] == a[:-1]
    conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
    return float(np.mean(same | conn))


def dedup(seq):
    out = []
    for x in seq:
        if x is None or x < 0:
            continue
        if not out or out[-1] != x:
            out.append(int(x))
    return out


def route_set_metrics(pred_arcs, ref_route, arc_len):
    """Sequence-level performance: length-weighted precision / recall / F1 of the
    set of arcs in the predicted route vs. the reference route."""
    P, R = set(dedup(pred_arcs)), set(int(x) for x in ref_route)
    inter = P & R
    lp = sum(arc_len[a] for a in P); lr = sum(arc_len[a] for a in R); li = sum(arc_len[a] for a in inter)
    prec = li / lp if lp > 0 else 0.0
    rec = li / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(route_precision=prec, route_recall=rec, route_f1=f1)
