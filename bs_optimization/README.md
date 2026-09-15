# Copper Production Optimizer

### ACCESS THE OPTIMIZER LIVE ONLINE HERE ###

https://copper-optimizer.fly.dev/

Internally developed physics-based optimization platform for copper extraction processes, combining advanced process modeling with economic analysis to maximize project value.

## Overview

This web application optimizes operational parameters across five critical stages of copper production:
- **Grinding** - Particle size reduction
- **Flotation** - Mineral concentration
- **Leaching** - Copper extraction
- **Solvent Extraction** - Solution purification
- **Electrowinning** - Copper recovery

The optimizer evaluates 22+ process parameters simultaneously to minimize costs while meeting production targets, providing complete technoeconomic analysis including NPV, IRR, and payback period calculations.

## Key Features

### Interactive Optimization
- Adjust feed ore grade (0.2% - 1% Cu)
- Set concentrate grade targets (15% - 35% Cu)
- Configure testing particle size (50-150 microns)
- Define target copper production (1,000 - 100,000 tons/year)

### Comprehensive Results
- Net Present Value (NPV) comparison
- Internal Rate of Return (IRR)
- Payback period analysis
- Optimal operating parameters for all process stages
- Stage-by-stage recovery rates
- Capital and operating cost breakdowns
- Cost per ton of copper produced

### Electrolyzer Integration
The platform automatically compares conventional operations with electrolyzer-enhanced processes, quantifying the economic impact of modern electrolysis technology.

## Technology Stack

- **Backend**: Flask (Python)
- **Optimization**: SciPy differential evolution
- **Process Models**: Physics-based equations for each unit operation
- **Cost Analysis**: Industry-standard CAPEX/OPEX models

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```

Access the application at `http://localhost:5000`

## File Structure

```
├── app.py                          # Flask web application
├── optimal_tea.py                  # Optimization engine
├── process_cost_integration.py     # Cost models
├── flotation_recovery.py           # Flotation physics
├── leaching.py                     # Leaching kinetics
├── solvent_extraction.py           # SX equilibrium
├── electrowinning.py               # EW electrochemistry
├── electrolyzer.py                 # Electrolyzer analysis
├── grind_break.py                  # Grinding models
├── tea.py                          # Economic analysis
├── templates/
│   └── index.html                  # Web interface
├── static/
│   ├── blueshift_logo.png
│   └── fsd_optimization_model.jpg
└── requirements.txt
```

## Performance

- Optimization typically completes in 30-60 seconds
- Uses differential evolution with 30 generations by default
- Evaluates thousands of parameter combinations to find global optimum

## Assumptions

- Copper price: $13,000/ton
- Mine life: 25 years
- Discount rate: 8%
- Concentrate grade: Variable (15-35% Cu)