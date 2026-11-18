# HMM + Viterbi route reconstruction: guide

Code: `src/pathsnap/hmm.py` (decoder), `scripts/05_hmm_eval.py` (tuning + evaluation)
Config: `configs/hmm_params.json`. Results: `results/hmm_vs_classifier_student_test.csv`, `results/hmm_grid.csv`

---

## 1. Why an HMM?
The classifier scores each GPS fix **on its own**. If you just take the best candidate for each fix, the route jumps:
- onto a parallel street for one fix,
- or onto the opposite direction of a two-way road,
- so consecutive predicted arcs are often **not connected**.

The HMM adds **sequence information + road connectivity**. It picks the most probable **connected** chain of arcs over the whole trip.

## 2. The model

| HMM part | Our implementation |
|---|---|
| **Observation** at time t | GPS fix t (x, y in metres, timestamp) |
| **Hidden state** at time t | One candidate of fix t: an (arc_id, offset along the arc) pair, i.e. *where the car is on the road*. Up to 8 states per fix (top 8 by classifier probability). |
| **Emission** P(fix ∣ state) | The classifier probability of that candidate, renormalised over the fix's candidates: `log e = w_e · log(p / Σp)` |
| **Transition** P(state_t ∣ state_t-1) | Newson & Krumm (2009) style. `log A = −|r − g| / β`. **r** is the *driving distance* on the network from the previous projection to the new one, found with a bounded Dijkstra on the **directed** graph. **g** is the *straight-line* distance between the two GPS fixes. A real move has r ≈ g. A jump to a parallel street needs a long detour, so r ≫ g, which makes the transition very unlikely. |
| U-turn penalty | Going from arc a straight onto its reverse arc costs an extra −2 in log-probability |
| Unreachable | If there is no path within the search limit, the transition is −∞ |

**Driving distance r** (`_route_dists` in hmm.py):
- On the same arc, moving forward: `r = offset_b − offset_a`. Backward moves up to 15 m count as GPS jitter, e.g. a stopped car.
- Otherwise: `r = (rest of arc a) + shortest path(end node of a → start node of b) + offset_b`.
- The Dijkstra search limit is `min(3000 m, 2g + 50·Δt + 50)`, so it grows with the time gap. Search results are cached per node for speed.

## 3. Viterbi recurrence
```
δ_1(j) = e_1(j)
δ_t(j) = e_t(j) + max_i [ δ_{t-1}(i) + log A(i → j) ]
ψ_t(j) = argmax_i [ ... ]              (back-pointer)
```
At the end, take `argmax δ_T` and follow the back-pointers ψ to recover one arc per fix. All of this is done in **log space**, so there is no numerical underflow over thousands of fixes.

Code layout:
- `transition_tables()` computes the r and g matrices **once per trip**, because they don't depend on β or w_e.
- `decode()` runs the Viterbi recurrence and back-tracking. It is fast, so tuning β and w_e is cheap.
- `viterbi()` is the wrapper that the API uses.

## 4. Edge cases (likely viva questions)
| Situation | What we do |
|---|---|
| **Fix with no candidates** (no road within 50 m) | No state. The prediction is −1 and the fix is skipped. The next transition spans the gap, with a bigger Dijkstra limit (it grows with Δt). |
| **Large GPS time gap** | Same idea: the search limit grows with elapsed time (50 m/s · Δt). |
| **All transitions infeasible** (consecutive fixes on disconnected segments) | **Chain break.** The chain restarts at that fix from its emissions alone, and each segment is back-tracked separately. One bad fix can't ruin the whole route. There were only 7 breaks across 19,160 student-test fixes. |
| **Stationary car** | Heading is unreliable. The 15 m backward tolerance stops jitter from forcing a break. |

## 5. Tuning (no leakage)
- β and w_e were grid-searched: β ∈ {2, 5, 10, 20, 40, 80} m and w_e ∈ {0.25, 0.5, 1, 2}.
- The search ran on **8 training trips** using **out-of-fold** classifier probabilities (from cross-validation), so the probabilities are as uncertain as they would be on unseen data.
- **Best: β = 5 m, w_e = 0.25.** A small β means transitions must match the GPS movement closely. A small w_e means the decoder relies more on road connectivity than on the classifier.
- The student-test trips were **not** used for tuning. The evaluator trips were never touched.

## 6. Results: classifier only vs classifier + HMM (10 student-test trips)
| Metric | Classifier only | + HMM |
|---|---|---|
| Observation Top-1 accuracy | 0.902 | **0.926** |
| Route continuity (consecutive arcs connected) | 0.953 | **0.979** |
| Route F1 (length-weighted arc overlap with reference) | 0.956 | **0.983** |

Per trip, the HMM improves **9 of 10** trips. For example, trip 00000077 goes from 0.786 to 0.829 and trip 00000076 from 0.937 to 0.979.

**Where it did NOT help: trip 00000007**, where Top-1 drops from 0.969 to 0.890. The car is parked at the end of the trip for about 150 fixes on a two-way street. With no movement there is no heading, and both directions are equally close, so the decoder chooses the reverse arc for that parked stretch. Route F1 still went up slightly (0.989 → 0.991).

Plots: `results/figures/hmm_vs_classifier.png` and `results/figures/hmm_grid.png`.
Maps (reference / classifier / HMM layers): `maps/student_test_<trip>.html`.

## 7. One-minute speaking version
> "The classifier picks a road for each GPS point on its own, which gives a broken route. We model the trip as an HMM. The hidden states are candidate road positions. The emission is the classifier's probability. The transition compares the driving distance on the road network with the straight-line GPS distance, so moves that need a detour, like jumping to a parallel street, get a low probability. Viterbi finds the most likely connected sequence. Fixes with no candidates are skipped, and if no transition is possible we break the chain and restart. We tuned β and the emission weight on training trips using out-of-fold probabilities. On student-test the HMM improved Top-1 from 0.902 to 0.926 and route continuity from 0.953 to 0.979. It only hurt one trip, where the car was parked on a two-way road."
