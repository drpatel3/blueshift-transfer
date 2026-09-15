"""Bound-desaturation probe.

Runs the inner DE search at a fixed reference scenario and reports where each
decision variable lands relative to its bounds. Used to track the
bound-desaturation campaign (see BOUND_DESATURATION_LOG.md).

    python -m process_model._bound_probe            # seed 42, 60x12
    python -m process_model._bound_probe 7 100 15   # seed 7, maxiter 100, popsize 15

Reads DECISION_VARS dynamically, so it keeps working as variables are added to
or removed from the DOE.
"""
from __future__ import annotations

import sys

from .topology import Topology
from .optimizer import Optimizer, DECISION_VARS

# Reference scenario = the current best-case winner topology (see
# information_flow_2.docx): 2000 tph, 0.8% Cu chalcopyrite, sulfide
# regrind(tower)+cleaner(2-stage) concentrator.
REF_TOPO = Topology(
    flotation_enabled=True, thickener_enabled=True, filter_enabled=True,
    ball_mill_enabled=True, cyclone_enabled=True, screen_enabled=True,
    regrind_enabled=True, cleaner_enabled=True,
    regrind_mill_type="tower", n_cleaner_stages=2,
)


def probe(maxiter: int = 60, popsize: int = 12, seed: int = 42,
          feed_tph: float = 2000.0, feed_grade: float = 0.008):
    opt = Optimizer(feed_tph=feed_tph, feed_grade=feed_grade)
    tr = opt.run_one(REF_TOPO, maxiter=maxiter, popsize=popsize, seed=seed)
    rows = []
    n_bound = 0
    for i, (name, lo, hi) in enumerate(DECISION_VARS):
        v = float(tr.best_x[i])
        frac = (v - lo) / (hi - lo)
        if frac >= 0.98:
            pos = "MAX"
        elif frac <= 0.02:
            pos = "MIN"
        else:
            pos = f"interior {frac*100:.0f}%"
        if pos in ("MAX", "MIN"):
            n_bound += 1
        rows.append((name, v, lo, hi, frac, pos))
    return tr, rows, n_bound


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    seed = int(argv[0]) if len(argv) > 0 else 42
    maxiter = int(argv[1]) if len(argv) > 1 else 60
    popsize = int(argv[2]) if len(argv) > 2 else 12
    tr, rows, n_bound = probe(maxiter=maxiter, popsize=popsize, seed=seed)
    print(f"# DOE bound probe - ref winner @ 2000tph/0.8%Cu, "
          f"seed={seed}, {maxiter}x{popsize}")
    print(f"{'variable':26s} {'value':>9s}  {'bounds':>13s}  position")
    print("-" * 64)
    for name, v, lo, hi, frac, pos in rows:
        b = f"[{lo:g},{hi:g}]"
        print(f"{name:26s} {v:9.3f}  {b:>13s}  {pos}")
    print("-" * 64)
    print(f"--> {n_bound}/{len(rows)} variables pinned to a bound")
    irr = f"{tr.irr*100:.0f}%" if tr.irr is not None else "n/a"
    print(f"NPV=${tr.best_npv/1e9:.2f}B  IRR={irr}  "
          f"capex=${tr.capex/1e6:.0f}M  R={tr.cu_recovery*100:.1f}%  "
          f"conc={tr.conc_grade_pct:.1f}%")


if __name__ == "__main__":
    main()
