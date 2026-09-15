# Stage Vocabulary V2 — Mid-Level Process Stages

Preserves engineering-meaningful distinctions that V1 collapsed.
Each stage has a unique role in the process sequence.

## Comminution (size reduction)
- **primary_crusher** — jaw/gyratory, first size reduction (maps from: crusher when order=1, primary_crusher, jaw_crusher, gyratory_crusher)
- **secondary_crusher** — cone crusher, second stage (maps from: crusher when order>1, cone_crusher, secondary_crusher)
- **hpgr** — high-pressure grinding rolls (maps from: hpgr)
- **sag_mill** — semi-autogenous grinding (maps from: sag_milling, sag_mill)
- **ball_mill** — ball mill grinding (maps from: ball_milling, ball_mill, milling, mill)
- **regrind** — secondary grinding for liberation (maps from: regrind, regrind_mill)
- **screen** — size classification through mesh (maps from: screen, vibrating_screen)
- **cyclone** — hydrocyclone classification (maps from: cyclone, hydrocyclone, classification)

## Concentration
- **rougher_flotation** — initial flotation recovery (maps from: flotation when first instance, rougher)
- **cleaner_flotation** — concentrate cleaning (maps from: flotation when after regrind, cleaner)
- **scavenger_flotation** — tails scavenging (maps from: scavenger)
- **gravity_concentration** — gravity separation (maps from: gravity, gravity_concentration, dms)
- **magnetic_separation** — magnetic separation (maps from: magnetic_separation)

## Hydrometallurgy
- **leach** — dissolution (heap or tank) (maps from: leach, heap_leach, tank_leach)
- **adsorption** — carbon/resin adsorption CIC/CIP/CIL (maps from: adsorption, cip, cic, cil)
- **elution** — carbon/resin stripping (maps from: elution, stripping, acid_wash, elution_regeneration)
- **solvent_extraction** — SX extraction (maps from: solvent_extraction, sx, sx_extraction)
- **electrowinning** — electrochemical metal recovery (maps from: electrowinning, cell)
- **precipitation** — chemical precipitation (maps from: precipitation, merrill_crowe, cementation)
- **ccd** — counter-current decantation (maps from: ccd, decantation)

## Pyrometallurgy
- **kiln** — high-temperature processing (maps from: kiln, roasting, calciner, reactor, autoclave)
- **smelting** — smelting furnace (maps from: smelting, furnace)
- **carbon_regeneration** — carbon kiln regeneration (maps from: regeneration, carbon_regeneration)

## Solid-Liquid Separation
- **concentrate_thickener** — concentrate dewatering (maps from: thickener when on concentrate path)
- **tailings_thickener** — tailings dewatering (maps from: thickener when on tails path)
- **filter** — filtration (maps from: filter, filter_press, vacuum_filter)

## Material Handling
- **input** — ROM feed (maps from: input, rom_stockpile)
- **stockpile** — ore storage (maps from: stockpile, ore_stockpile)
- **bin** — storage containers (maps from: bin, fine_ore_bin, concentrate_bin)
- **feeder** — controlled discharge (maps from: feeder, apron_feeder, hopper)
- **conveyor** — belt transport (maps from: conveyor)

## Product / Waste
- **concentrate_product** — saleable concentrate (maps from: concentrate, concentrate_product, product, dore_product, cathode_product)
- **tailing** — tailings disposal (maps from: tailing, tailings, pond, tailing_disposal)
- **solution_pond** — PLS/barren ponds (maps from: solution_ponds, pond when in leach circuit)
- **water_treatment** — effluent/cyanide treatment (maps from: water_treatment, cyanide_destruction, detox, neutralization)

## Auxiliary
- **tank** — conditioning/storage vessels (maps from: tank, conditioning_tank, agitation_tank)
- **agglomeration** — ore agglomeration for heap leach (maps from: agglomeration)

---

**Total: ~37 stages** (vs 23 in V1, vs 1545 raw)

Key distinctions preserved:
1. primary_crusher vs secondary_crusher (sequence matters)
2. sag_mill vs ball_mill (different circuit positions)
3. rougher_flotation vs cleaner_flotation (the core of the flotation circuit)
4. concentrate_thickener vs tailings_thickener (parallel paths)
5. elution vs adsorption (sequential in carbon circuit)
6. solvent_extraction as distinct from precipitation
7. gravity_concentration as a real stage (32 docs)
8. carbon_regeneration separate from kiln
