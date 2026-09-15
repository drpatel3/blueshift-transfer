"""Phase 1 topology-selecting NPV optimizer for the Cu-sulfide flowsheet.

Outer loop: enumerate `topology.all_topologies()`.
Inner loop: `scipy.optimize.differential_evolution` over the continuous
operating params per topology (see `DECISION_VARS`).
Objective: maximize `tea.evaluate(...)['npv']` at fixed 8% discount rate.

Decision-variable bounds are user-specified realistic operating ranges
(see plan no-lets-stop-here-sharded-mango.md). The DOE membership is being
refined (see BOUND_DESATURATION_LOG.md) to remove non-design variables and
restore the trade-offs that keep the rest off their bounds.

Pattern mirrors `optimization_model/.../optimal_tea.py:673-698`.

CLI:
    python -m process_model.optimizer
"""
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution

from .throughput import simulate_cu_sulfide, CuSulfideParams
from .tea import evaluate, TEAParams
from .topology import Topology, all_topologies, archetypes, topology_label


# 10-D decision vector. Order matters — `_apply()` decodes by index.
DECISION_VARS: tuple[tuple[str, float, float], ...] = (
    ("sp_power",                0.8,  1.2),    # kW/m^3
    # Upper bound widened 1.5 -> 2.2 cm/s (bound-desaturation Attempt 4) so
    # the froth-flooding rollover region [1.6, 2.2] is inside the search space.
    # Practical TankCell Jg ceiling ~2.0-2.2 cm/s (Wills 7th ed. Ch. 12).
    ("sp_gas_rate",             0.8,  2.2),    # cm/s
    ("frother_conc",           10.0, 50.0),    # ppm
    ("flot_residence_time_min", 15.0, 20.0),    # min, bank-total
    ("air_fraction",            0.10, 0.25),
    ("slurry_fraction",         0.15, 0.30),
    # contact_angle removed from the DOE (bound-desaturation Attempt 2): it is a
    # mineral-surface PROPERTY (chalcopyrite-xanthate wettability), not a design
    # dial. NPV rises monotonically in it up to the floatability plateau, so DE
    # rode it to whatever bound was set. Now fixed at 55 deg (the validated
    # plateau value; see throughput.py CuSulfideParams.flot_op).
    # flot_cell_volume_m3 removed from optimizer: cell volume is a discrete
    # equipment selection (FLSmidth/Outotec TankCell e500 = 500 m^3), not a
    # continuous design variable. Set on CuSulfideParams.flot_cell_volume_m3.
    # circulating_load removed from DOE (Attempt 6): CL affects BM kW but
    # the flotation-feed F80 is set by target_flot_P80_um independently. The
    # classification-sharpness benefit (higher CL → better liberation) can't
    # be represented without touching the kinetic model. Fixed at 2.5 (industry
    # standard; Wills 7th ed. Ch. 5) in CuSulfideParams.circulating_load.
    # tailings_adverse_factor was removed from the DOE (bound-desaturation
    # Attempt 1): it is a TSF *site condition*, not a design choice, so the
    # optimizer only ever drove it to its 1.0 floor for free NPV. It is now a
    # fixed scenario input (TEAParams.tailings_adverse_factor, default 1.0);
    # set it per project for an unfavorable site.
    # Flotation feed P80 — drives the recovery vs grinding-power trade-off.
    # Cascades through: BM Bond Ecs (more kW for finer grind),
    # cyclone O/F & flotation kinetics. Bounds = industrial Cu rougher band
    # (Wills' Mineral Processing Technology, 7th ed., Ch. 12).
    ("target_flot_P80_um",     75.0, 200.0),   # um
)
BOUNDS: list[tuple[float, float]] = [(lo, hi) for _, lo, hi in DECISION_VARS]
NAMES: list[str] = [name for name, _, _ in DECISION_VARS]


def _apply(x: np.ndarray, base: CuSulfideParams,
           tea_base: TEAParams) -> tuple[CuSulfideParams, TEAParams]:
    """Decode a decision vector into mutated CuSulfideParams + TEAParams.

    Mapping is by NAME (via `NAMES`), not by index: a variable that is dropped
    from `DECISION_VARS` simply falls back to its base-params default, so the
    DOE membership can change without re-threading positional indices. `x` must
    align with the current `DECISION_VARS` order.
    """
    d = {name: float(v) for name, v in zip(NAMES, x)}

    def pick(name, fallback):
        return d.get(name, fallback)

    fo = base.flot_op
    flot_op = replace(
        fo,
        sp_power=pick("sp_power", fo.sp_power),
        sp_gas_rate=pick("sp_gas_rate", fo.sp_gas_rate),
        frother_conc=pick("frother_conc", fo.frother_conc),
        air_fraction=pick("air_fraction", fo.air_fraction),
        slurry_fraction=pick("slurry_fraction", fo.slurry_fraction),
        contact_angle=pick("contact_angle", fo.contact_angle),
    )
    p = replace(base,
                flot_op=flot_op,
                flot_residence_time_min=pick("flot_residence_time_min",
                                             base.flot_residence_time_min),
                circulating_load=pick("circulating_load", base.circulating_load),
                target_flot_P80_um=pick("target_flot_P80_um",
                                        base.target_flot_P80_um))
    t = replace(tea_base,
                tailings_adverse_factor=pick("tailings_adverse_factor",
                                             tea_base.tailings_adverse_factor))
    return p, t


_FAILURE_PENALTY = 1.0e12  # large positive (DE minimizes -NPV)


def _objective(x: np.ndarray, *, base: CuSulfideParams, tea_base: TEAParams,
               topo: Topology, feed_tph: float, feed_grade: float) -> float:
    """-NPV with finite penalty for simulator failures."""
    try:
        p, t = _apply(x, base, tea_base)
        sim = simulate_cu_sulfide(feed_tph, feed_grade, p, topology=topo)
        res = evaluate(sim, t)
        npv = res["npv"]
        if not np.isfinite(npv):
            return _FAILURE_PENALTY
        # Mild penalty for degenerate-recovery regions so DE doesn't get
        # parked in obviously-broken corners
        R = sim["balance"]["overall_cu_recovery"]
        if R < 0.05:
            return _FAILURE_PENALTY
        return -float(npv)
    except Exception:
        return _FAILURE_PENALTY


@dataclass
class TopologyResult:
    topology: Topology
    label: str
    best_x: np.ndarray
    best_npv: float
    irr: Optional[float]
    capex: float
    cu_recovery: float
    conc_grade_pct: float
    de_message: str


@dataclass
class OptimizationResult:
    feed_tph: float
    feed_grade: float
    results: list[TopologyResult] = field(default_factory=list)

    @property
    def winner(self) -> TopologyResult:
        return max(self.results, key=lambda r: r.best_npv)


@dataclass
class Optimizer:
    base_params: CuSulfideParams = field(default_factory=CuSulfideParams)
    tea_params: TEAParams = field(default_factory=TEAParams)
    feed_tph: float = 2000.0
    feed_grade: float = 0.008

    def run_one(self, topo: Topology, *, maxiter: int = 200, popsize: int = 15,
                seed: int = 42, workers: int = 1, tol: float = 0.01,
                disp: bool = False) -> TopologyResult:
        kwargs = dict(base=self.base_params, tea_base=self.tea_params,
                      topo=topo, feed_tph=self.feed_tph,
                      feed_grade=self.feed_grade)
        de = differential_evolution(
            lambda x: _objective(x, **kwargs),
            BOUNDS,
            maxiter=maxiter, popsize=popsize, tol=tol, atol=0.0,
            seed=seed, workers=workers, disp=disp, polish=True,
        )
        # Final evaluation at the optimum to pull TEA & sim diagnostics
        p, t = _apply(de.x, self.base_params, self.tea_params)
        sim = simulate_cu_sulfide(self.feed_tph, self.feed_grade, p, topology=topo)
        res = evaluate(sim, t)
        return TopologyResult(
            topology=topo,
            label=topology_label(topo),
            best_x=np.array(de.x, dtype=float),
            best_npv=float(res["npv"]),
            irr=res["irr"] if res["irr"] == res["irr"] else None,
            capex=float(res["total_capex"]),
            cu_recovery=float(sim["balance"]["overall_cu_recovery"]),
            conc_grade_pct=float(sim["balance"]["conc_grade_pct"]),
            de_message=str(de.message),
        )

    def run(self, topologies: Optional[list[Topology]] = None,
            maxiter: int = 200, popsize: int = 15, seed: int = 42,
            workers: int = 1, tol: float = 0.01,
            disp: bool = False) -> OptimizationResult:
        # Default to the curated archetype set (~12 topologies). Pass
        # `topologies=all_topologies()` explicitly for the full 500+ sweep.
        topos = topologies if topologies is not None else archetypes()
        out = OptimizationResult(feed_tph=self.feed_tph,
                                 feed_grade=self.feed_grade)
        for topo in topos:
            tr = self.run_one(topo, maxiter=maxiter, popsize=popsize,
                              seed=seed, workers=workers, tol=tol, disp=disp)
            out.results.append(tr)
        # Sort high-NPV first for convenient inspection
        out.results.sort(key=lambda r: r.best_npv, reverse=True)
        return out


def print_leaderboard(opt_result: OptimizationResult) -> None:
    print("=== Topology leaderboard (sorted by NPV) ===")
    print(f"  feed: {opt_result.feed_tph:.0f} tph @ {opt_result.feed_grade*100:.2f}% Cu")
    print()
    header = (f"  {'rank':>4}  {'topology':<35s}  {'NPV ($B)':>9s}  "
              f"{'IRR':>7s}  {'capex ($M)':>11s}  {'R_Cu':>6s}  {'grade':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(opt_result.results, 1):
        irr_s = f"{r.irr*100:6.1f}%" if r.irr is not None else "    n/a"
        print(f"  {i:>4d}  {r.label:<35s}  {r.best_npv/1e9:>9.2f}  "
              f"{irr_s}  {r.capex/1e6:>11.1f}  "
              f"{r.cu_recovery*100:>5.1f}% {r.conc_grade_pct:>5.1f}%")
    print()
    win = opt_result.winner
    print("=== Winner — best continuous params ===")
    print(f"  topology: {win.label}")
    for name, val in zip(NAMES, win.best_x):
        print(f"    {name:<25s} = {val:.4f}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    opt = Optimizer()
    # Conservative defaults — still ~1-3 min on a laptop, far less when
    # workers=-1 is allowed
    res = opt.run(maxiter=80, popsize=12, workers=1, disp=False)
    print_leaderboard(res)
    print(f"\n(elapsed: {time.time()-t0:.1f} s)")
