# Fiber Tracking Analysis

A Python-based tool for analyzing fiber skeleton images and computing shape metrics including curvature, length, and branching patterns.

## Features

- Skeleton image processing and graph construction
- Branch point detection and pruning
- Curvature analysis along fiber paths
- Component-wise shape analysis
- Visualization of results

## Dependencies

Dependencies can be found in `requirements.txt` file in the root directory.

## Installation

```bash
git clone https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis.git
cd Confocal-Rheology-Fiber-Detection-and-Analysis/
pip install -r requirements.txt
```

## Usage

### Input Files Required

- `skeleton.tif`: Binary skeleton image sequence (TIFF stack)
- `fiber.tif`: Original fiber image sequence (TIFF stack) for overlay visualization

### Running the Analysis

```bash
python calculate_component_metrics.py skeleton.tif fiber.tif
```

### Output

- Frame-by-frame metrics in `frame_metrics_2.csv`
- Visualization plots saved in `Plots/` directory:
  - Pruned skeleton overlays
  - Curvature analysis plots
  - Component visualizations

## Example Output

The script generates:
- CSV files with quantitative metrics per frame
- Plots showing the skeleton analysis process
- Visualization of branch points and centerline detection

## Output CSV Format
The analysis generates `frame_metrics.csv` with the following structure:

| Column | Description |
|--------|-------------|
| Frame | Frame number from image sequence |
| Component ID | Unique identifier for each fiber component |
| Total Length | Fiber component length in pixels |
| Average Slope | Mean slope of the fiber component |
| Average Curvature | Mean curvature measurement |

Example:
```csv
Frame,Component ID,Total Length,Average Slope,Average Curvature
1,1,106.013009848,2.312378464,0.179883812
1,2,5.950000000,-33063582739.333300,1.162197e-09
