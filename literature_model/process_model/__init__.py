"""Process models for flowsheet stages, built from literature equations.

crusher.py    — Pothina 2007 (gyratory, Bond + amperage constant)
sag.py        — Asghari 2019 (Bond Wi + JK drop-weight)
hydrocyclone.py — Samaeili 2017 / Plitt 1976 (cut size, pressure, efficiency)
thickener.py  — Galvez 2014 (Richardson-Zaki, Arterburn, cost)
sx.py         — Moreno 2009 (mixer-settler dynamic ODEs)
flotation.py  — Abrahamson/Luttrell-Yoon/Finch-Dobby (mechanical-cell recovery)
flowsheet.py  — steady-state chain of stages by mass balance
throughput.py — reference Cu-sulfide simulator (calls flowsheet stages)
tea.py        — BlueShift TEA + O'Hara 1992 SME Ch. 6.3 capex equations
topology.py   — flowsheet on/off decisions for the optimizer
optimizer.py  — Phase 1 NPV optimizer (outer enum + scipy DE)
"""

from .topology import Topology, all_topologies, topology_label
from .optimizer import Optimizer, OptimizationResult, TopologyResult

__all__ = [
    "Topology", "all_topologies", "topology_label",
    "Optimizer", "OptimizationResult", "TopologyResult",
]
