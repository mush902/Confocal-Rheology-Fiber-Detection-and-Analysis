# Agarose Elongation Analysis
A Python-based tool for analyzing fiber orientation and length distribution in skeletal fiber networks over time.

## Features
- Constructs graph representations from binary skeleton images
- Detects and eliminates spurious diagonal connections using cycle detection
- Partitions fibers into components for detailed analysis
- Calculates key metrics:
  - Contour length of fiber segments
  - Begin-to-end distances
  - Fiber orientation angles
- Generates temporal distribution visualizations:
  - Heatmaps of angle distribution changes over time
  - 3D surface plots of fiber length vs angle vs time
  - Time-lapse animations of angle distributions

## Dependencies
Dependencies can be found in the `requirements.txt` file in the root directory.

## Installation
```bash
git clone https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis.git
cd Confocal-Rheology-Fiber-Detection-and-Analysis/Agarose_Elongation/
pip install -r requirements.txt
```

## Usage
### Input Files Required
- `skeleton.tif`: Binary skeleton image sequence (TIFF stack)

### Running the Analysis
```bash
python analyzing_angle_length_distribution.py skeleton.tif [bin_size] [component_length_threshold]
```

### Optional Parameters
- `bin_size`: Angular bin size in degrees (default: 5)
- `component_length_threshold`: Minimum fiber segment length to include in analysis (default: 10)

### Example Command
```bash
python analyzing_angle_length_distribution.py InputData/RegionRightTop/skeleton.tif 5 10
```

## Output
The script creates an output folder named `fiber_analysis_output/` containing:
- `skeleton_matrix.csv`: Raw data matrix of angle distribution over frames
- `skeleton_heatmap_final.png`: Heatmap visualization showing angle distribution evolution
- `skeleton_evolution.mp4`: Animation of angle distribution changes across frames
- `skeleton_3d_surface_plot.png`: 3D surface plot of fiber length vs angle vs time

## Output CSV Format
### `skeleton_matrix.csv`
| Column | Description |
|--------|-------------|
| frame | Frame number in the sequence |
| 0.0-5.0, 5.0-10.0, etc. | Average length of fibers in each angle bin |

Example:
```csv
frame,0.0-5.0,5.0-10.0,...,175.0-180.0
1,150.8,201.6,...,179.2
10,145.2,210.3,...,168.7
...
```

## Example Plot
<img src="https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis/blob/main/Agarose_Elongation/fiber_analysis_output/RegionRightTop/skeleton_3d_surface_plot.png" width="400"/>


## Directory Structure
```
Agarose_Elongation/
├── analyzing_angle_length_distribution.py
├── InputData/
│   └── RegionRightTop/
│       ├── skeleton.tif
│       ├── binarised.tif
│       └── colored.tif
└── fiber_analysis_output/
    └── RegionRightTop/
        ├── skeleton_matrix.csv
        ├── skeleton_heatmap_final.png
        ├── skeleton_evolution.mp4
        └── skeleton_3d_surface_plot.png
```
