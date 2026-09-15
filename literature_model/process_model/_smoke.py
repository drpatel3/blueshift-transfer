"""Quick end-to-end sanity check: ROM feed -> crusher -> SAG -> cyclone -> thickener."""
from process_model import crusher, sag, hydrocyclone as hc, thickener as th
from process_model.flowsheet import (Stream, run_crusher, run_sag,
                                      run_hydrocyclone, run_thickener)


def main():
    feed = Stream(solids_tph=800.0, water_tph=0.0, F80_um=150_000.0,
                  cu_grade=0.0086)

    cp = crusher.CrusherParams(Wi=14.0)
    after_crush, kw_crush = run_crusher(feed, cp, P80_um=25_000.0)
    print(f"Crusher: P80={after_crush.F80_um:.0f} um, power={kw_crush:.1f} kW")

    # add water for SAG feed slurry (~68% solids)
    sag_feed = Stream(solids_tph=after_crush.solids_tph,
                      water_tph=after_crush.solids_tph * 0.47,
                      F80_um=after_crush.F80_um, cu_grade=after_crush.cu_grade)

    sp = sag.SAGParams(A=67.62, b=1.0, Wi=11.28, ta=0.67, PLI=47.0)
    after_sag, kw_sag = run_sag(sag_feed, sp, Ecs_kwh_per_t=8.0)
    print(f"SAG:     P80={after_sag.F80_um:.0f} um, power={kw_sag:.1f} kW")

    # Cyclone
    geom = hc.CycloneGeom(Dc=50.0, Di=15.0, Dx=20.0, Du=10.0, h=100.0)
    uf, of, diag = run_hydrocyclone(after_sag, geom, pressure_kpa=69.0)
    print(f"Cyclone: d50={diag['d50_um']:.1f} um, dP={diag['dp_kpa']:.1f} kPa, "
          f"Rv={diag['Rv']:.2f}, solids_to_UF={diag['solids_to_uf']:.2f}")
    print(f"  UF: {uf.solids_tph:.0f} t/h solids, {uf.water_tph:.0f} t/h water ({uf.solids_pct:.1f}%)")
    print(f"  OF: {of.solids_tph:.0f} t/h solids, {of.water_tph:.0f} t/h water ({of.solids_pct:.1f}%)")

    # Thickener on cyclone OF (tailings water recovery)
    thp = th.ThickenerParams(vTF=2.5, rF=0.25, n=4.0)
    tuf, tof, tdiag = run_thickener(of, thp, target_uf_pct=50.0)
    print(f"Thickener: area={tdiag['area_m2']:.1f} m2, D={tdiag['diameter_m']:.1f} m")
    print(f"  Dewatered UF: {tuf.solids_pct:.1f}% solids, clear water {tof.water_tph:.0f} t/h")


if __name__ == "__main__":
    main()
