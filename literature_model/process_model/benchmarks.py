"""Benchmark harness for the three success metrics in PROGRESS.md.

Run any of:
    python -m process_model.benchmarks --m3a
    python -m process_model.benchmarks --m3b [--quick|--full]
    python -m process_model.benchmarks --m1
    python -m process_model.benchmarks --m2
    python -m process_model.benchmarks --all

Each metric returns a structured dict and prints a markdown-ready table.
The four readings can be pasted under each metric's "Current reading:" line
in `process_model/PROGRESS.md`.

Notes
- M3a is fast (closed-form sims, 30 runs).
- M3b runs DE; --quick is the dev path, --full is the recorded reading.
- M1/M2 are reporting-only (no pass/fail asserts) — failures here drive
  follow-up plans, not test reds.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Optional

from .topology import Topology
from .throughput import CuSulfideParams, simulate_cu_sulfide
from .tea import TEAParams, evaluate
from .leach import LeachParams
from .heap_leach import HeapLeachParams


# ----------------------------------------------------------------- routes
# Canonical Cu sulfide plant: rougher + regrind + 3-stage cleaner bank.
# Per-stage R_g = 0.50 with N=3 gives R_g_bank ~ 0.125 (Cobre Panama 25-26%,
# Escondida 28-30% disclosures; Wills 7th ed. Ch. 12.4-12.5).
DEFAULT_SULFIDE = Topology(regrind_enabled=True, cleaner_enabled=True,
                           n_cleaner_stages=3)
CONC_POX = Topology(filter_enabled=False, regrind_enabled=True,
                    cleaner_enabled=True, n_cleaner_stages=3,
                    leach_enabled=True,
                    sx_enabled=True, ew_enabled=True)
HEAP = Topology(flotation_enabled=False, ball_mill_enabled=False,
                cyclone_enabled=False, thickener_enabled=False,
                filter_enabled=False, heap_leach_enabled=True,
                sx_enabled=True, ew_enabled=True)


def _classify_route(t: Topology) -> str:
    if t.heap_leach_enabled:
        return "heap"
    if t.leach_enabled:
        return "POX"
    return "sulfide"


# ============================================================== M3a closure
def run_m3a_closure(throughput_tph: float = 2000.0,
                    grades: tuple[float, ...] = tuple(
                        round(g, 4) for g in
                        [0.002, 0.004, 0.006, 0.008, 0.010,
                         0.012, 0.014, 0.016, 0.018, 0.020]),
                    verbose: bool = True) -> dict:
    """For each (grade, route), |cu_closure_err| / cu_in_tph. Pass < 0.001."""
    routes = {"sulfide": DEFAULT_SULFIDE, "POX": CONC_POX, "heap": HEAP}
    rows: list[dict] = []
    for g in grades:
        for name, topo in routes.items():
            params = _params_for_route(name)
            sim = simulate_cu_sulfide(throughput_tph, g, params, topology=topo)
            bal = sim["balance"]
            cu_in = bal["cu_in_tph"]
            err = abs(bal["cu_closure_err"])
            ratio = err / cu_in if cu_in > 0 else 0.0
            rows.append({"grade": g, "route": name,
                         "cu_in_tph": cu_in, "abs_err_tph": err,
                         "ratio": ratio})
    max_ratio = max(r["ratio"] for r in rows)
    out = {"rows": rows, "max_ratio": max_ratio,
           "pass": max_ratio < 0.001}
    if verbose:
        _print_m3a(out)
    return out


def _print_m3a(out: dict) -> None:
    print("### M3a — Mass-balance closure\n")
    print(f"| grade  | route   | cu_in (t/h) | |err| (t/h) | ratio |")
    print(f"|--------|---------|-------------|-------------|-------|")
    for r in out["rows"]:
        print(f"| {r['grade']*100:5.2f}% | {r['route']:<7s} "
              f"| {r['cu_in_tph']:11.4f} | {r['abs_err_tph']:11.3e} "
              f"| {r['ratio']:.2e} |")
    flag = "PASS" if out["pass"] else "FAIL"
    print(f"\nmax ratio = {out['max_ratio']:.2e} ({flag}; threshold 1.0e-03)\n")


# ============================================================ M3b DE stability
def run_m3b_de_stability(grades: tuple[float, ...] = (0.004, 0.010, 0.018),
                         seeds: tuple[int, ...] = (42, 137, 271, 404, 1729),
                         throughput_tph: float = 2000.0,
                         maxiter: int = 200, popsize: int = 15,
                         workers: int = 1,
                         verbose: bool = True) -> dict:
    """5-seed NPV spread per grade on the default sulfide topology.

    Pass: (max-min)/|mean| <= 0.05 at every grade.
    """
    from .optimizer import Optimizer  # local import — heavy scipy dependency
    rows: list[dict] = []
    for g in grades:
        opt = Optimizer(feed_tph=throughput_tph, feed_grade=g)
        npvs: list[float] = []
        for s in seeds:
            t0 = time.time()
            tr = opt.run_one(DEFAULT_SULFIDE, maxiter=maxiter,
                             popsize=popsize, seed=s, workers=workers,
                             tol=0.01, disp=False)
            npvs.append(tr.best_npv)
            if verbose:
                print(f"  grade={g*100:.2f}%  seed={s:>4d}  "
                      f"NPV=${tr.best_npv/1e9:6.3f}B  "
                      f"({time.time()-t0:.1f}s)")
        mean = sum(npvs) / len(npvs)
        spread = (max(npvs) - min(npvs)) / abs(mean) if mean != 0 else float("inf")
        rows.append({"grade": g, "npvs": npvs, "mean": mean,
                     "spread": spread})
    max_spread = max(r["spread"] for r in rows)
    out = {"rows": rows, "max_spread": max_spread,
           "pass": max_spread <= 0.05,
           "maxiter": maxiter, "popsize": popsize, "seeds": list(seeds)}
    if verbose:
        _print_m3b(out)
    return out


def _print_m3b(out: dict) -> None:
    print("\n### M3b — DE convergence stability\n")
    print(f"seeds={out['seeds']}, maxiter={out['maxiter']}, "
          f"popsize={out['popsize']}\n")
    print(f"| grade  | mean NPV ($B) | min ($B) | max ($B) | spread |")
    print(f"|--------|---------------|----------|----------|--------|")
    for r in out["rows"]:
        print(f"| {r['grade']*100:5.2f}% | {r['mean']/1e9:12.3f}  "
              f"| {min(r['npvs'])/1e9:8.3f} "
              f"| {max(r['npvs'])/1e9:8.3f} | {r['spread']*100:5.2f}% |")
    flag = "PASS" if out["pass"] else "FAIL"
    print(f"\nmax spread = {out['max_spread']*100:.2f}% "
          f"({flag}; threshold 5.00%)\n")


# ============================================================ M1 route selection
@dataclass
class M1Project:
    name: str
    grade: float            # head grade fraction
    mineralogy: str         # 'chalcopyrite' | 'chalcocite' | 'oxide' | 'IOCG'
    actual_route: str       # 'sulfide' | 'POX' | 'heap'
    throughput_tph: float = 2000.0
    # Disclosed byproduct head grades. None => fall back to the
    # mineralogy default in throughput.BYPRODUCT_GRADES.
    mo_grade_ppm: Optional[float] = None
    au_g_per_t:   Optional[float] = None
    ag_g_per_t:   Optional[float] = None
    # Disclosed Cu-concentrate As grade (percent). None => fall back to the
    # mineralogy default in throughput.AS_CONC_PCT_BY_MINERALOGY.
    conc_arsenic_pct: Optional[float] = None


# Byproduct head grades from public reserve / production disclosures:
#   Cobre Panama   — FQM 2019 NI 43-101 TR (Cu 0.40%, Mo 0.014%, Au 0.07 g/t,
#                    Ag 1.6 g/t for the Botija + Colina deposits).
#   Cerro Verde    — Freeport-McMoRan 2024 10-K (Cu 0.36%, Mo 0.019%,
#                    Au n/d (trace), Ag 2.8 g/t).
#   QB2            — Teck 2018 NI 43-101 TR (Cu 0.51%, Mo 0.020%,
#                    Au n/d, Ag 1.5 g/t).
#   Cobre Antamina — Glencore 2024 ASR (Cu 0.92%, Mo 0.024%, Ag 12 g/t —
#                    elevated by the Cu-Zn skarn endmember).
#   Olympic Dam    — BHP 2024 ASR (Cu 0.90%, Au 0.50 g/t, Ag 1.7 g/t,
#                    no Mo, U/REE outside concentrator scope).
#   Heap-route plants (Spence/Lomas Bayas/Mantoverde/Radomiro Tomic) and
#   El Abra POX leave byproduct fields as None — defaults apply.
M1_PROJECTS: tuple[M1Project, ...] = (
    M1Project("Cobre Panama (FQM)", 0.004, "chalcopyrite", "sulfide",
              mo_grade_ppm=140.0, au_g_per_t=0.07, ag_g_per_t=1.6),
    M1Project("Cerro Verde (Freeport)", 0.004, "chalcopyrite", "sulfide",
              mo_grade_ppm=190.0, au_g_per_t=0.04, ag_g_per_t=2.8),
    M1Project("Quebrada Blanca QB2 (Teck)", 0.005, "chalcopyrite", "sulfide",
              mo_grade_ppm=200.0, au_g_per_t=0.04, ag_g_per_t=1.5),
    M1Project("Cobre Antamina (Glencore)", 0.009, "chalcopyrite", "sulfide",
              mo_grade_ppm=240.0, au_g_per_t=0.10, ag_g_per_t=12.0),
    M1Project("BHP Spence", 0.005, "chalcocite", "heap"),
    M1Project("Lomas Bayas (Glencore)", 0.003, "chalcocite", "heap"),
    M1Project("Mantoverde (Capstone)", 0.005, "chalcocite/oxide", "heap"),
    M1Project("Codelco Radomiro Tomic", 0.004, "oxide", "heap"),
    M1Project("Phelps Dodge El Abra", 0.006, "chalcocite", "POX"),
    M1Project("Olympic Dam (BHP)", 0.009, "IOCG", "sulfide",
              mo_grade_ppm=0.0, au_g_per_t=0.50, ag_g_per_t=1.7),
)


def _params_for_route(route_name: str,
                      mineralogy: Optional[str] = None,
                      mo_grade_ppm: Optional[float] = None,
                      au_g_per_t: Optional[float] = None,
                      ag_g_per_t: Optional[float] = None,
                      conc_arsenic_pct: Optional[float] = None,
                      ) -> CuSulfideParams:
    """Build CuSulfideParams with mineralogy set. The simulator
    (`simulate_cu_sulfide`) reads `MINERALOGY_PROPS` and overrides
    `leach_params.max_extraction` + `heap_leach_params.extraction_fraction`
    + the rougher flotation cap automatically — no manual override needed.

    Byproduct grades default to the mineralogy table when None.
    """
    return replace(CuSulfideParams(),
                   mineralogy=(mineralogy or "chalcopyrite"),
                   mo_grade_ppm=mo_grade_ppm,
                   au_g_per_t=au_g_per_t,
                   ag_g_per_t=ag_g_per_t,
                   conc_arsenic_pct=conc_arsenic_pct)


def run_m1_route_selection(verbose: bool = True) -> dict:
    """For each of the 10 disclosed projects, enumerate the FULL
    `topology.archetypes()` set (every valid stage-by-stage flowsheet)
    and pick the NPV winner. Classify by route only after winning.

    The model is choosing the full topology start to finish — the route
    label is the consequence of the chosen topology, not a pre-fixed
    triple. Accuracy vs disclosed plants takes a hit; that is the
    intentional trade.
    """
    from .topology import archetypes, topology_label
    archs = archetypes()
    tea = TEAParams()
    rows: list[dict] = []
    for proj in M1_PROJECTS:
        params = _params_for_route("any", proj.mineralogy,
                                   mo_grade_ppm=proj.mo_grade_ppm,
                                   au_g_per_t=proj.au_g_per_t,
                                   ag_g_per_t=proj.ag_g_per_t,
                                   conc_arsenic_pct=proj.conc_arsenic_pct)
        best_npv = float("-inf")
        best_topo = None
        for topo in archs:
            try:
                sim = simulate_cu_sulfide(proj.throughput_tph, proj.grade,
                                          params, topology=topo)
                res = evaluate(sim, tea)
                if res["npv"] > best_npv:
                    best_npv = res["npv"]
                    best_topo = topo
            except Exception:
                continue
        winner = _classify_route(best_topo) if best_topo else "n/a"
        label = topology_label(best_topo) if best_topo else "n/a"
        match = (winner == proj.actual_route)
        rows.append({"project": proj.name, "grade": proj.grade,
                     "mineralogy": proj.mineralogy,
                     "actual": proj.actual_route, "predicted": winner,
                     "match": match, "topology_label": label,
                     "best_npv": best_npv})
    matches = sum(1 for r in rows if r["match"])
    total = len(rows)
    out = {"rows": rows, "matches": matches, "total": total,
           "score": matches / total,
           "pass": matches / total >= 0.80}
    if verbose:
        _print_m1(out)
    return out


def _print_m1(out: dict) -> None:
    print("\n### M1 — Route-selection accuracy\n")
    print(f"Model selects the full topology end-to-end "
          f"(via topology.archetypes()); route label is the consequence.\n")
    print(f"| project                          | grade  | mineralogy        "
          f"| actual  | predicted | hit | chosen topology |")
    print(f"|----------------------------------|--------|-------------------"
          f"|---------|-----------|-----|-----------------|")
    for r in out["rows"]:
        print(f"| {r['project']:<32s} | {r['grade']*100:5.2f}% "
              f"| {r['mineralogy']:<17s} | {r['actual']:<7s} "
              f"| {r['predicted']:<9s} | {'Y' if r['match'] else 'N':<3s} "
              f"| {r['topology_label']} |")
    flag = "PASS" if out["pass"] else "FAIL"
    print(f"\nscore = {out['matches']}/{out['total']} "
          f"({out['score']*100:.0f}%) ({flag}; threshold 80%)\n")


# ============================================================== M2 metallurgy
@dataclass
class M2Plant:
    name: str
    route: str               # 'sulfide' | 'POX' | 'heap'
    grade: float             # head grade fraction
    mineralogy: str
    reported_R_cu: float     # reported overall Cu recovery, fraction
    reported_conc_grade_pct: Optional[float] = None  # sulfide only


# Reference R_Cu definitions vary by route:
#   sulfide : ore-to-concentrate Cu recovery (the conventional "plant R_Cu")
#   POX     : concentrate-to-cathode efficiency (leach × SX × EW; what
#             POX project disclosures cite — Sherritt 96% etc.)
#   heap    : ore-to-cathode Cu recovery (the conventional heap R_Cu)
M2_PLANTS: tuple[M2Plant, ...] = (
    # sulfide — ore-to-conc
    M2Plant("Cobre Panama", "sulfide", 0.004, "chalcopyrite", 0.87, 25.5),
    M2Plant("Escondida concentrator", "sulfide", 0.008, "chalcopyrite", 0.87, 29.0),
    M2Plant("Cerro Verde", "sulfide", 0.004, "chalcopyrite", 0.86, 30.0),
    # concentrate-POX — conc-to-cathode (leach × SX × EW)
    M2Plant("El Abra POX", "POX", 0.006, "chalcocite", 0.95),
    M2Plant("Sherritt Bagdad pilot", "POX", 0.004, "chalcopyrite", 0.96),
    M2Plant("Morenci POX (Freeport)", "POX", 0.004, "chalcopyrite", 0.94),
    # heap — ore-to-cathode
    M2Plant("Spence heap", "heap", 0.005, "chalcocite", 0.73),
    M2Plant("Lomas Bayas heap", "heap", 0.003, "chalcocite", 0.70),
    M2Plant("Mantoverde heap", "heap", 0.005, "chalcocite/oxide", 0.75),
)


def run_m2_metallurgy(verbose: bool = True) -> dict:
    routes = {"sulfide": DEFAULT_SULFIDE, "POX": CONC_POX, "heap": HEAP}
    rows: list[dict] = []
    for pl in M2_PLANTS:
        topo = routes[pl.route]
        params = _params_for_route(pl.route, pl.mineralogy)
        sim = simulate_cu_sulfide(2000.0, pl.grade, params, topology=topo)
        bal = sim["balance"]
        conc_grade = bal["conc_grade_pct"]
        if pl.route == "POX":
            # POX disclosures are concentrate-to-cathode (leach×SX×EW), not
            # ore-to-cathode. Compute the same metric on the model side.
            cu_in_conc = (sim["streams"]["filter_cake"].solids_tph
                          * sim["streams"]["filter_cake"].cu_grade)
            cathode = bal["cu_in_cathode_tph"]
            R_model = cathode / cu_in_conc if cu_in_conc > 0 else 0.0
        else:
            R_model = bal["overall_cu_recovery"]
        dR = abs(R_model - pl.reported_R_cu) * 100.0  # percentage points
        dG = (abs(conc_grade - pl.reported_conc_grade_pct)
              if pl.reported_conc_grade_pct is not None else None)
        rows.append({"plant": pl.name, "route": pl.route, "grade": pl.grade,
                     "R_plant": pl.reported_R_cu, "R_model": R_model,
                     "dR_pp": dR,
                     "conc_plant": pl.reported_conc_grade_pct,
                     "conc_model": conc_grade, "dG_pp": dG})
    median_dR = median(r["dR_pp"] for r in rows)
    sulfide_rows = [r for r in rows if r["route"] == "sulfide"
                    and r["dG_pp"] is not None]
    median_dG = median(r["dG_pp"] for r in sulfide_rows) if sulfide_rows else None
    out = {"rows": rows, "median_dR_pp": median_dR,
           "median_dG_pp": median_dG,
           "pass_R": median_dR <= 5.0,
           "pass_G": (median_dG is not None and median_dG <= 4.0)}
    if verbose:
        _print_m2(out)
    return out


def _print_m2(out: dict) -> None:
    print("\n### M2 — Recovery + concentrate-grade agreement\n")
    print(f"| plant                       | route   | grade  | R_plant "
          f"| R_model | dR_pp | conc_plant | conc_model | dG_pp |")
    print(f"|-----------------------------|---------|--------|---------"
          f"|---------|-------|------------|------------|-------|")
    for r in out["rows"]:
        cp = (f"{r['conc_plant']:5.1f}%" if r['conc_plant'] is not None
              else "  n/a")
        cm = (f"{r['conc_model']:5.1f}%" if r['route'] == 'sulfide'
              else "  n/a")
        dg = f"{r['dG_pp']:5.2f}" if r['dG_pp'] is not None else "  n/a"
        print(f"| {r['plant']:<27s} | {r['route']:<7s} "
              f"| {r['grade']*100:5.2f}% | {r['R_plant']*100:5.1f}% "
              f"| {r['R_model']*100:5.1f}% | {r['dR_pp']:5.2f} "
              f"| {cp:<10s} | {cm:<10s} | {dg} |")
    flag_R = "PASS" if out["pass_R"] else "FAIL"
    flag_G = "PASS" if out["pass_G"] else "FAIL"
    dG_str = (f"{out['median_dG_pp']:.2f}" if out['median_dG_pp'] is not None
              else "n/a")
    print(f"\nmedian |dR| = {out['median_dR_pp']:.2f}pp ({flag_R}; threshold 5pp)")
    print(f"median |dG| = {dG_str}pp ({flag_G}; threshold 4pp)\n")


# ============================================================== orchestrator
def run_all(m3b_full: bool = False) -> dict:
    print("=" * 70)
    print("process_model success-metric benchmarks")
    print("=" * 70)
    res_m3a = run_m3a_closure()
    res_m1 = run_m1_route_selection()
    res_m2 = run_m2_metallurgy()
    if m3b_full:
        res_m3b = run_m3b_de_stability(maxiter=200, popsize=15)
    else:
        res_m3b = run_m3b_de_stability(maxiter=20, popsize=8)
    return {"m3a": res_m3a, "m3b": res_m3b, "m1": res_m1, "m2": res_m2}


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3a", action="store_true")
    ap.add_argument("--m3b", action="store_true")
    ap.add_argument("--m1", action="store_true")
    ap.add_argument("--m2", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="m3b only: maxiter=20, popsize=8 (~5 min)")
    ap.add_argument("--full", action="store_true",
                    help="m3b only: maxiter=200, popsize=15 (~30-40 min)")
    args = ap.parse_args(argv)
    if args.all:
        run_all(m3b_full=args.full)
        return
    if args.m3a:
        run_m3a_closure()
    if args.m1:
        run_m1_route_selection()
    if args.m2:
        run_m2_metallurgy()
    if args.m3b:
        if args.full:
            run_m3b_de_stability(maxiter=200, popsize=15)
        else:
            run_m3b_de_stability(maxiter=20, popsize=8)


if __name__ == "__main__":
    main()
