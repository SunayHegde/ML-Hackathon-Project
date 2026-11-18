"""HMM + Viterbi route reconstruction.

Hidden state at fix t : one candidate (arc, offset-along-arc) of fix t, i.e. the
                        vehicle's position on the road network at time t.
Emission              : classifier probability of the candidate, renormalised
                        over the fix's candidates:  w_e * log p_norm.
Transition (a -> b)   : Newson & Krumm (2009) style. r = network driving distance
                        from the projection on arc a to the projection on arc b
                        (bounded Dijkstra on the DIRECTED graph), g = straight-line
                        distance between the two fixes.
                            log P = -|r - g| / beta  (- uturn_penalty for a->reverse(a))
                        Unreachable within the search bound -> -inf.
Gaps / breaks         : fixes without candidates get no state (prediction -1) and
                        the next transition spans the gap with a search bound that
                        grows with elapsed time. If every transition into fix t is
                        infeasible (consecutive fixes suggest disconnected road
                        segments), the chain is broken and restarted at t from its
                        emissions alone ("break"), so one bad fix cannot derail
                        the whole route.
Decoding              : Viterbi recurrence
        delta_t(j) = e_t(j) + max_i [ delta_{t-1}(i) + log A_{t-1,t}(i, j) ]
                        with back-pointers; back-tracking segment by segment.

Transition distance tables (r, g) do not depend on beta / w_e, so they are
computed once per trip (`transition_tables`) and re-used when tuning.
"""
from dataclasses import dataclass, asdict
import numpy as np

BACKWARD_TOL_M = 15.0


@dataclass
class HMMParams:
    beta: float = 20.0          # transition scale (m)
    w_emission: float = 1.0     # weight on log emission
    uturn_penalty: float = 2.0  # extra log-penalty for a -> reverse(a)
    max_cutoff: float = 3000.0  # max Dijkstra search radius (m)
    max_states: int = 8         # keep the top-N candidates (by classifier prob) per fix

    def to_dict(self):
        return asdict(self)


def prune(cands, probs, max_states):
    out_c, out_p = [], []
    for (a, d, o), p in zip(cands, probs):
        p = np.asarray(p, dtype=np.float64)
        if len(a) > max_states:
            k = np.argsort(-p, kind="stable")[:max_states]
            a, d, o, p = a[k], d[k], o[k], p[k]
        out_c.append((a, d, o)); out_p.append(p)
    return out_c, out_p


def _route_dists(net, a, oa, b_ids, b_off, cutoff):
    """Driving distance from (arc a, offset oa) to each (b_ids[v], b_off[v])."""
    out = np.full(len(b_ids), np.inf)
    rem = net.length[a] - oa
    sp = None
    if rem <= cutoff:
        sp = net.bounded_sp(int(net.arcs[a, 1]), cutoff - rem)
    for v, (b, ob) in enumerate(zip(b_ids, b_off)):
        if a == b and ob >= oa - BACKWARD_TOL_M:
            # forward along the same arc; small backward moves are GPS jitter
            # (e.g. a stopped vehicle), counted as their absolute length
            out[v] = abs(ob - oa)
        elif sp is not None:
            d = sp.get(int(net.arcs[b, 0]))
            if d is not None:
                out[v] = rem + d + ob
    return out


def transition_tables(net, xy, t, cands, params: HMMParams):
    valid = [i for i in range(len(cands)) if len(cands[i][0]) > 0]
    tables = {}
    for prev, i in zip(valid[:-1], valid[1:]):
        a_ids, _, a_off = cands[prev]
        b_ids, _, b_off = cands[i]
        g = float(np.linalg.norm(xy[i] - xy[prev]))
        dt = max(float(t[i] - t[prev]), 1.0)
        cutoff = min(params.max_cutoff, 2.0 * g + 50.0 * dt + 50.0)
        R = np.vstack([_route_dists(net, int(a_ids[u]), float(a_off[u]), b_ids, b_off, cutoff)
                       for u in range(len(a_ids))])
        fa, ta = net.arcs[a_ids, 0][:, None], net.arcs[a_ids, 1][:, None]
        fb, tb = net.arcs[b_ids, 0][None, :], net.arcs[b_ids, 1][None, :]
        uturn = (fa == tb) & (ta == fb) & (a_ids[:, None] != b_ids[None, :])
        tables[i] = (prev, R, g, uturn)
    return valid, tables


def decode(cands, probs, valid, tables, params: HMMParams):
    T = len(cands)
    pred = np.full(T, -1, dtype=np.int64)
    if not valid:
        return pred, dict(breaks=0)

    def emis(i):
        p = probs[i] + 1e-9
        return params.w_emission * np.log(p / p.sum())

    deltas, backs = {valid[0]: emis(valid[0])}, {valid[0]: None}
    breaks = 0
    for i in valid[1:]:
        prev, R, g, uturn = tables[i]
        with np.errstate(invalid="ignore"):
            logA = -np.abs(R - g) / params.beta - params.uturn_penalty * uturn
        logA[~np.isfinite(R)] = -np.inf
        scores = deltas[prev][:, None] + logA
        best = np.argmax(scores, axis=0)
        val = scores[best, np.arange(scores.shape[1])]
        e = emis(i)
        if not np.isfinite(val).any():
            breaks += 1
            deltas[i], backs[i] = e, None
        else:
            deltas[i], backs[i] = val + e, (prev, best)

    pos = {v: k for k, v in enumerate(valid)}
    i = valid[-1]
    s = int(np.argmax(deltas[i]))
    while True:
        pred[i] = cands[i][0][s]
        bp = backs[i]
        if bp is None:
            k = pos[i]
            if k == 0:
                break
            i = valid[k - 1]
            s = int(np.argmax(deltas[i]))
        else:
            i, best = bp[0], bp[1]
            s = int(best[s])
    return pred, dict(breaks=breaks)


def viterbi(net, xy, t, cands, probs, params: HMMParams = HMMParams()):
    """Full decode. Returns predicted arc per fix (-1 = no candidates) + diagnostics."""
    cands, probs = prune(cands, probs, params.max_states)
    valid, tables = transition_tables(net, xy, t, cands, params)
    pred, diag = decode(cands, probs, valid, tables, params)
    diag["n_states"] = int(sum(len(c[0]) for c in cands))
    return pred, diag
