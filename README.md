# Confocal Rheology Fiber Detection and Analysis Toolkit

A suite of Python tools for quantitative analysis of fiber networks in confocal microscopy images.

## Installation
```bash
git clone https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis.git
cd Confocal-Rheology-Fiber-Detection-and-Analysis
```

## Dependencies

- Python (>= 3.x)
  
All module dependencies are listed in `requirements.txt`. 

Install with:
```bash
pip install -r requirements.txt
```

## Modules Overview

### 1. SDC Fiber Bending Analysis
- Location: `SDC_Fiber_Bending/`
- Purpose: Track fiber bending angles and deformation over time
- Key Features: Angle measurement at fixed lengths, linear fitting of bending rates

### 2. Fiber Diameter Analysis  
- Location: `Diameter_Analysis/`
- Purpose: Measure fiber diameters through perpendicular cross-sections  
- Key Features: Midpoint detection, component-level statistics, network-wide averages

### 3. Agarose Elongation Analysis
- Location: `Agarose_Elongation/`  
- Purpose: Quantify fiber orientation and length distributions  
- Key Features: Temporal heatmaps, 3D surface plots, time-lapse animations

### 4. Agarose Connectivity Analysis  
- Location: `Agarose_Connectivity_Junction_Density/`  
- Purpose: Analyze network connectivity through junction detection  
- Key Features: Degree distribution calculation, junction classification
