"""Quick test: apply V2 stage mapping to all graphs and count edges."""
import json, boto3
from collections import defaultdict

V2_DIRECT_MAP = {
    "sag_milling": "sag_mill", "sag_mill": "sag_mill", "sag": "sag_mill",
    "ball_milling": "ball_mill", "ball_mill": "ball_mill",
    "rod_milling": "ball_mill", "rod_mill": "ball_mill",
    "milling": "ball_mill", "grinding": "ball_mill",
    "hpgr": "hpgr", "regrind": "regrind", "regrind_mill": "regrind",
    "screen": "screen", "vibrating_screen": "screen", "screening": "screen", "grizzly": "screen",
    "cyclone": "cyclone", "hydrocyclone": "cyclone", "classification": "cyclone", "ciclones": "cyclone",
    "gravity": "gravity_concentration", "gravity_concentration": "gravity_concentration",
    "dms": "gravity_concentration", "dms_plant": "gravity_concentration",
    "magnetic_separation": "magnetic_separation",
    "leach": "leach", "leaching": "leach", "heap_leach": "leach", "tank_leaching": "leach",
    "adsorption": "adsorption", "cip": "adsorption", "cic": "adsorption", "cil": "adsorption",
    "elution": "elution", "stripping": "elution", "acid_wash": "elution",
    "elution_regeneration": "elution", "elution_and_electrowinning": "elution",
    "solvent_extraction": "solvent_extraction", "sx": "solvent_extraction", "sx_extraction": "solvent_extraction",
    "electrowinning": "electrowinning", "cell": "electrowinning",
    "precipitation": "precipitation", "merrill_crowe": "precipitation", "cementation": "precipitation",
    "ion_exchange": "ion_exchange", "ix": "ion_exchange",
    "ccd": "ccd", "decantation": "ccd",
    "kiln": "kiln", "roasting": "kiln", "calciner": "kiln", "reactor": "kiln",
    "autoclave": "kiln", "pressure_oxidation": "kiln",
    "smelting": "smelting", "furnace": "smelting",
    "regeneration": "carbon_regeneration", "carbon_regeneration": "carbon_regeneration",
    "filter": "filter", "filter_press": "filter", "filtration": "filter", "dewatering": "filter",
    "input": "input", "rom_stockpile": "input",
    "stockpile": "stockpile", "bin": "bin", "feeder": "feeder", "hopper": "feeder",
    "conveyor": "conveyor", "tailing": "tailing", "tailings": "tailing", "pond": "tailing",
    "detox": "water_treatment", "water_treatment": "water_treatment",
    "cyanide_destruction": "water_treatment", "neutralization": "water_treatment",
    "solution_ponds": "solution_pond",
    "dore": "concentrate_product", "dore_product": "concentrate_product",
    "concentrate": "concentrate_product", "product": "concentrate_product",
    "cathode_product": "concentrate_product", "refinery": "concentrate_product",
    "refining": "concentrate_product", "gold_room": "concentrate_product",
    "tank": "tank", "conditioning": "tank", "agglomeration": "agglomeration",
    "crusher": "_crusher", "crushing": "_crusher",
    "flotation": "_flotation", "thickener": "_thickener", "thickening": "_thickener",
    "mill": "_mill",
}

REMOVABLE = {"copper", "cu", "gold", "au", "iron", "fe", "lead", "pb", "zinc", "zn",
             "moly", "silver", "polymetallic", "pyrite"}

def map_v2(raw_id, order=0.5, in_raws=None, out_raws=None):
    sid = raw_id.lower().strip()
    tokens = [t for t in sid.split("_") if t not in REMOVABLE]
    if not tokens:
        return None
    sid_clean = "_".join(tokens)

    v2 = V2_DIRECT_MAP.get(sid) or V2_DIRECT_MAP.get(sid_clean)
    if not v2:
        for t in reversed(tokens):
            if t in V2_DIRECT_MAP:
                v2 = V2_DIRECT_MAP[t]
                break
    if not v2:
        return None

    in_raws = in_raws or []
    out_raws = out_raws or []

    if v2 == "_crusher":
        v2 = "primary_crusher" if order < 0.3 else "secondary_crusher"
    elif v2 == "_mill":
        v2 = "sag_mill" if order < 0.4 else "ball_mill"
    elif v2 == "_flotation":
        v2 = "cleaner_flotation" if any("regrind" in r for r in in_raws) else "rougher_flotation"
    elif v2 == "_thickener":
        if any(r in ("tailing", "tailings", "pond", "disposal") for r in out_raws):
            v2 = "tailings_thickener"
        elif any(r in ("filter", "bin", "output", "concentrate") for r in out_raws):
            v2 = "concentrate_thickener"
        else:
            v2 = "tailings_thickener" if order > 0.7 else "concentrate_thickener"

    # Override with specific raw names
    if "rougher" in sid:
        v2 = "rougher_flotation"
    if "cleaner" in sid or "scavenger" in sid or "column_cleaning" in sid:
        v2 = "cleaner_flotation"
    if "sag" in sid and v2 in ("ball_mill", "_mill"):
        v2 = "sag_mill"

    return v2


s3 = boto3.client("s3")
bucket = "sagemaker-us-east-1-666109694894"
edge_counts = defaultdict(int)
edge_docs = defaultdict(set)
unmapped = defaultdict(int)
total_docs = 0

for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="graphs/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json") or any(x in key for x in ["cross_doc", "vocab", "summary"]):
            continue
        try:
            g = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            doc_id = g.get("document_id", "")
            nodes = {}
            for n in g.get("nodes", []):
                if n.get("type") == "stage":
                    sid = n.get("stage_id", n["id"].replace("stg_", ""))
                    order = n.get("features", {}).get("order_normalized", 0.5)
                    nodes[n["id"]] = {"raw": sid, "order": order}
            if not nodes:
                continue
            total_docs += 1

            out_by = defaultdict(list)
            in_by = defaultdict(list)
            for e in g.get("edges", []):
                if e.get("type") != "stage_transition":
                    continue
                src_raw = nodes.get(e["source"], {}).get("raw", "")
                dst_raw = nodes.get(e["target"], {}).get("raw", "")
                out_by[e["source"]].append(dst_raw)
                in_by[e["target"]].append(src_raw)

            v2_map = {}
            for nid, info in nodes.items():
                v2 = map_v2(info["raw"], info["order"], in_by.get(nid, []), out_by.get(nid, []))
                if v2:
                    v2_map[nid] = v2
                else:
                    unmapped[info["raw"]] += 1

            for e in g.get("edges", []):
                if e.get("type") != "stage_transition":
                    continue
                sv = v2_map.get(e["source"])
                dv = v2_map.get(e["target"])
                if sv and dv and sv != dv:
                    edge_counts[f"{sv}|{dv}"] += 1
                    edge_docs[f"{sv}|{dv}"].add(doc_id)
        except Exception:
            pass

print(f"Docs: {total_docs}")
print(f"Unique V2 edges: {len(edge_counts)}")
print(f"Edges >=3 docs: {sum(1 for v in edge_docs.values() if len(v) >= 3)}")
print()
print("Top 40 V2 edges:")
for key, count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:40]:
    s, d = key.split("|")
    nd = len(edge_docs[key])
    print(f"  {s:25s} -> {d:25s}  count={count:4d}  docs={nd:3d}")
print()
print("Top unmapped:")
for r, c in sorted(unmapped.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {r:30s}  {c}")
