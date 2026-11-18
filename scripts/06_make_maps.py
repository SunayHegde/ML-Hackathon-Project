"""Stage 6: Folium maps for the student-testing trips.

Layers: raw GPS fixes, road network (candidate arcs), reference route,
classifier-only prediction, classifier + HMM prediction; layer control + legend.
Requires predictions cached by 05_hmm_eval.py (cache/pred_<trip>.npy).

Usage: python scripts/06_make_maps.py
Output: maps/student_test_<trip>.html
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from pathsnap import config as C
from pathsnap.features import prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.viz import route_map


def main():
    for tid in list_trips(C.STUDENT_TEST_DIR):
        raw = load_trip(C.STUDENT_TEST_DIR, tid)
        td = prepare_trip(raw, with_labels=True)
        clf, hmm = np.load(C.CACHE_DIR / f"pred_{tid}.npy")
        acc_c, acc_h = np.mean(clf == td.label), np.mean(hmm == td.label)
        layers = {
            "Reference route (.route)": (list(raw.route), "#2e7d32", 8, None),
            f"Classifier-only prediction (Top-1 {acc_c:.3f})": (list(clf), "#ef6c00", 3, "6 6"),
            f"Classifier + HMM/Viterbi (Top-1 {acc_h:.3f})": (list(hmm), "#d81b60", 4, None),
        }
        m = route_map(td, layers, f"[student_test] trip {tid}")
        out = C.MAPS_DIR / f"student_test_{tid}.html"
        m.save(str(out))
        print("wrote", out)


if __name__ == "__main__":
    main()
