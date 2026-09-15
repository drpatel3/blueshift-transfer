"""Flowsheet topology — which optional stages exist in the chain.

Booleans gate stage on/off. Crusher, SAG, and rougher flotation are always
on (degenerate without them). The sulfide-flotation chain has four legacy
toggles (thickener, filter, ball mill, cyclone) plus three extensions
(vibrating screen, regrind, cleaner). The hydromet branch (atmospheric
leach -> SX -> EW + neutralization) treats rougher flotation tails when
its toggles are on; cathode product adds to concentrate revenue at the
cathode_payable_fraction (no TC/RC).

Stream routing for off-states is documented in the run-blocks of
`throughput.simulate_cu_sulfide`. This module describes the decision
space and labels, plus filters physically-degenerate combos.
"""
from dataclasses import dataclass, fields
from itertools import product


REGRIND_MILL_TYPES: tuple[str, ...] = ("ball", "tower", "isamill")


@dataclass(frozen=True)
class Topology:
    # sulfide flotation route
    flotation_enabled: bool = True
    thickener_enabled: bool = True
    filter_enabled: bool = True
    ball_mill_enabled: bool = True
    cyclone_enabled: bool = True
    screen_enabled: bool = False
    regrind_enabled: bool = False
    cleaner_enabled: bool = False
    # Regrind mill technology — discrete equipment-selection lever, only
    # consulted when regrind_enabled. Options: "ball" (conventional ball
    # mill, Wi ~ 14 kWh/t, baseline capex), "tower" (Metso Vertimill VTM,
    # ~25% lower kWh/t at fine P80, ~85% capex; Mazzinghy 2014, Jankovic
    # 2003), "isamill" (Glencore IsaMill M3000/M5000, ~40% lower kWh/t
    # below 25 µm, ~140% capex due to ceramic media + specialized
    # equipment; Burford & Clark 2007 AusIMM).
    regrind_mill_type: str = "ball"
    # Number of cleaner flotation stages in series. 1-3 typical for Cu.
    # Each stage rejects ~50% of the gangue carrying through; banks compound
    # geometrically (R_g_bank = R_g_per^N). Cu kinetic per stage assumed
    # full internal recycle (no per-stage Cu loss bookkept). Capex scales
    # linearly with N (one cell-bank per stage).
    # Reference: Wills 7th ed. Ch. 12.4 — multi-stage cleaner cascades;
    # Cobre Panama / Escondida disclosed flowsheets run 2-3 cleaner stages.
    # When cleaner_enabled=False, validity forces N=1 (canonical). Default
    # is 1 so that the bare Topology() (no cleaner) stays valid; archetypes
    # that enable the cleaner pass n_cleaner_stages=2 or 3 explicitly.
    n_cleaner_stages: int = 1
    # Concentrate-leach hydromet (POX/Albion) — feeds flotation concentrate
    # to a tank leach. Mutually exclusive with concentrate sale: when
    # leach_enabled, tea.py books cathode revenue only and zeros the
    # concentrate-sale line. Requires flotation_enabled.
    leach_enabled: bool = False
    # Whole-ore HEAP leach — primary-crushed ore stacked on a lined pad,
    # irrigated with H2SO4 PLS. Skips fine grinding and flotation entirely.
    # Suited to low-grade oxide / supergene chalcocite ore. Mutually
    # exclusive with both concentrate-leach and concentrate sale.
    heap_leach_enabled: bool = False
    sx_enabled: bool = False
    ew_enabled: bool = False
    neutralization_enabled: bool = False


def _is_valid(t: Topology) -> bool:
    """Filter physically-degenerate combinations.

    Three mutually-exclusive product routes:
    1. SULFIDE FLOTATION — concentrate sale to smelter (default).
    2. CONCENTRATE-LEACH HYDROMET — flotation feeds a tank leach
       (POX/Albion). No concentrate sale; cathode product only.
    3. WHOLE-ORE HEAP LEACH — primary-crushed ore on lined pad, no
       flotation. Cathode product only. Suited to low-grade oxide /
       supergene chalcocite ore.

    Rules:
    - flotation OR heap_leach must be on (one of the two routes must
      provide the Cu source).
    - flotation and heap_leach are mutually exclusive (a deposit goes
      to one or the other, not both).
    - leach (concentrate) requires flotation (leach feed = concentrate).
    - leach (concentrate) and heap_leach are mutually exclusive (different
      feed; would double-count Cu).
    - cyclone needs ball mill; cleaner needs regrind.
    - SX needs leach OR heap_leach; EW needs SX; neutralization needs
      either leach path.
    - Heap leach skips fine grinding: BM/cyclone/regrind/cleaner/thickener/
      filter must be off (they have no role in a heap-leach circuit).
    """
    # Must have one of the two Cu-source routes
    if not t.flotation_enabled and not t.heap_leach_enabled:
        return False
    # Mutually exclusive routes
    if t.flotation_enabled and t.heap_leach_enabled:
        return False
    if t.leach_enabled and t.heap_leach_enabled:
        return False
    # Sulfide-side dependencies
    if t.cyclone_enabled and not t.ball_mill_enabled:
        return False
    if t.cleaner_enabled and not t.regrind_enabled:
        return False
    # Concentrate-leach (POX/Albion) dependencies. Industrial autoclaves
    # run on CLEANER concentrate at ~25-30% Cu (Glencore Albion at McArthur
    # River, Phelps Dodge El Abra POX, Sherritt Bagdad pilot — all use
    # 3-stage cleaner banks ahead of the autoclave). Routing rougher froth
    # at 1-3% Cu directly to POX is not practiced — autoclave $/kg Cu
    # explodes once slurry volume per tonne of contained Cu blows up. Force
    # leach -> cleaner (and cleaner -> regrind already enforced above), so
    # any concentrate-leach archetype must build its conc grade first.
    if t.leach_enabled and not t.flotation_enabled:
        return False
    if t.leach_enabled and not t.cleaner_enabled:
        return False
    # Hydromet downstream chain — accept either leach path as upstream
    has_leach_path = t.leach_enabled or t.heap_leach_enabled
    if t.sx_enabled and not has_leach_path:
        return False
    if t.ew_enabled and not t.sx_enabled:
        return False
    if t.neutralization_enabled and not has_leach_path:
        return False
    # Heap leach excludes fine-grinding and dewatering stages
    if t.heap_leach_enabled:
        if t.ball_mill_enabled or t.cyclone_enabled:
            return False
        if t.regrind_enabled or t.cleaner_enabled:
            return False
        if t.thickener_enabled or t.filter_enabled:
            return False
    # Regrind mill type must be a known option. When regrind is OFF,
    # force the canonical "ball" default to avoid duplicate equivalent
    # topologies in the enumeration.
    if t.regrind_mill_type not in REGRIND_MILL_TYPES:
        return False
    if not t.regrind_enabled and t.regrind_mill_type != "ball":
        return False
    # Cleaner-stage count: 1-3 valid; force N=1 when cleaner is off to
    # avoid duplicate equivalent topologies in the enumeration.
    if t.n_cleaner_stages not in (1, 2, 3):
        return False
    if not t.cleaner_enabled and t.n_cleaner_stages != 1:
        return False
    return True


def all_topologies() -> list[Topology]:
    """Enumerate the on/off + discrete-mill-type space, drop invalid combos.

    11 booleans × 3 regrind-mill-types -> 6144 raw combos; validity rules
    cut it down considerably. Use `archetypes()` for the optimizer's
    default leaderboard sweep — running DE on the full set is slow.
    """
    bool_keys = [f.name for f in fields(Topology)
                 if f.name not in ("regrind_mill_type", "n_cleaner_stages")]
    out: list[Topology] = []
    for vals in product([True, False], repeat=len(bool_keys)):
        for mill_type in REGRIND_MILL_TYPES:
            for n_clean in (1, 2, 3):
                kwargs = dict(zip(bool_keys, vals))
                kwargs["regrind_mill_type"] = mill_type
                kwargs["n_cleaner_stages"] = n_clean
                t = Topology(**kwargs)
                if _is_valid(t):
                    out.append(t)
    return out


def archetypes() -> list[Topology]:
    """Curated shortlist of distinctive flowsheet archetypes for the
    optimizer's default leaderboard sweep. The two main routes a real Cu
    plant picks between:

        SULFIDE: flotation + (optional) regrind + cleaner + dewatering
                 — for high-grade Cu sulfide ores
        HYDROMET: leach + SX + EW + (optional) neutralization
                 — for oxide ores or low-grade where flotation can't justify
                   the upgrade economics

    Hybrid chain (concentrator + tails-leach) is included as a less-common
    third option but generally wins only at marginal Cu in tails.
    """
    A = []
    # ---- Sulfide route variants (flotation -> concentrate sale)
    # Standard rougher only
    A.append(Topology())
    # Standard with regrind + cleaner — one variant per regrind mill type.
    # Ball is baseline; tower (Vertimill) is ~25% lower kWh/t at fine P80
    # for ~85% capex; IsaMill is ~40% lower kWh/t below 25 µm for ~140% capex.
    for mt in REGRIND_MILL_TYPES:
        for n_clean in (1, 2, 3):
            A.append(Topology(regrind_enabled=True, cleaner_enabled=True,
                              regrind_mill_type=mt, n_cleaner_stages=n_clean))
        A.append(Topology(screen_enabled=True, regrind_enabled=True,
                          cleaner_enabled=True, regrind_mill_type=mt,
                          n_cleaner_stages=2))
    # Rougher only, no filter (truck wet conc — small operations)
    A.append(Topology(filter_enabled=False))
    # SAG-only grinding (small, hard ore)
    A.append(Topology(ball_mill_enabled=False, cyclone_enabled=False))
    # ---- Hydromet route variants (flotation -> concentrate -> leach -> SX -> EW)
    # Rougher + concentrate-leach + SX + EW (POX/Albion-style on bulk conc)
    A.append(Topology(filter_enabled=False,
                      leach_enabled=True, sx_enabled=True, ew_enabled=True))
    # Rougher + regrind + cleaner + concentrate-leach — variants per mill type
    for mt in REGRIND_MILL_TYPES:
        for n_clean in (1, 2, 3):
            A.append(Topology(filter_enabled=False,
                              regrind_enabled=True, cleaner_enabled=True,
                              leach_enabled=True, sx_enabled=True, ew_enabled=True,
                              regrind_mill_type=mt, n_cleaner_stages=n_clean))
    # Hydromet + neutralization (cleans up SX raffinate)
    A.append(Topology(filter_enabled=False,
                      regrind_enabled=True, cleaner_enabled=True,
                      leach_enabled=True, sx_enabled=True, ew_enabled=True,
                      neutralization_enabled=True))
    # ---- Whole-ore heap leach route (no flotation, no fine grinding)
    # Crusher only -> heap pad -> SX -> EW. Suits low-grade oxide /
    # supergene chalcocite at <0.5% Cu. Cheap capex, lower extraction.
    A.append(Topology(flotation_enabled=False, ball_mill_enabled=False,
                      cyclone_enabled=False, thickener_enabled=False,
                      filter_enabled=False,
                      heap_leach_enabled=True,
                      sx_enabled=True, ew_enabled=True))
    # Heap leach + neutralization for raffinate clean-up
    A.append(Topology(flotation_enabled=False, ball_mill_enabled=False,
                      cyclone_enabled=False, thickener_enabled=False,
                      filter_enabled=False,
                      heap_leach_enabled=True,
                      sx_enabled=True, ew_enabled=True,
                      neutralization_enabled=True))
    return [t for t in A if _is_valid(t)]


def topology_label(t: Topology) -> str:
    """Human-readable single-line description for leaderboards."""
    if t.heap_leach_enabled:
        # Heap-leach route — primary crush only, no fine grinding or flotation
        parts = ["crusher / heap"]
        hydromet = ["heap-leach"]
        if t.sx_enabled:
            hydromet.append("SX")
        if t.ew_enabled:
            hydromet.append("EW")
        if t.neutralization_enabled:
            hydromet.append("neut")
        parts.append("+".join(hydromet))
        return "[heap] " + " / ".join(parts)

    grind = "SAG+BM" if t.ball_mill_enabled else "SAG-only"
    if t.cyclone_enabled:
        grind += " closed"
    elif t.ball_mill_enabled:
        grind += " open"
    if t.screen_enabled:
        grind = "screen+" + grind

    parts = [grind]

    if t.flotation_enabled:
        flot = "rougher"
        if t.regrind_enabled:
            mt_label = {"ball": "regrind", "tower": "regrind(tower)",
                        "isamill": "regrind(IsaMill)"}.get(
                            t.regrind_mill_type, "regrind")
            flot += "+" + mt_label
        if t.cleaner_enabled:
            n = t.n_cleaner_stages
            flot += "+cleaner" if n == 1 else f"+cleaner({n}-stage)"
        dewater = []
        if t.thickener_enabled:
            dewater.append("thickener")
        if t.filter_enabled:
            dewater.append("filter")
        dewater_s = "+".join(dewater) if dewater else "no dewatering"
        parts.append(f"{flot} / {dewater_s}")

    if t.leach_enabled:
        hydromet = ["leach"]
        if t.sx_enabled:
            hydromet.append("SX")
        if t.ew_enabled:
            hydromet.append("EW")
        if t.neutralization_enabled:
            hydromet.append("neut")
        parts.append("+".join(hydromet))

    # Sulfide vs hydromet are mutually exclusive at the revenue level:
    # leach on -> concentrate goes to leach (cathode product, no conc sale).
    if t.leach_enabled:
        return "[hydromet] " + " / ".join(parts)
    return "[sulfide] " + " / ".join(parts)
