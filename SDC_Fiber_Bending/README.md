# SDC Fiber Bending Analysis

A Python-based tool for analyzing fiber skeleton images to detect and track bending angles of SDC fibers over time.

## Features

- Constructs a graph from binary skeleton images
- Detects and removes spurious diagonal connections through cycle detection
- Identifies leaf nodes in the network structure
- Calculates key metrics including contour length, height, width, and angles
- Tracks angle changes at fixed lengths along fibers over time
- Generates linear fits of angle change rates
- Visualizes results with customizable plots

## Dependencies

Dependencies can be found in the `requirements.txt` file in the root directory.

## Installation

```bash
git clone https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis.git
cd Confocal-Rheology-Fiber-Detection-and-Analysis/SDC_Fiber_Bending/
pip install -r requirements.txt
```

## Usage

### Input Files Required
- `skeleton.tif`: Binary skeleton image sequence (TIFF stack)
- `binarised.tif`: Binarized confocal fiber image sequence (TIFF stack) for visualization

### Running the Analysis

```bash
python track_sdc_angle.py skeleton.tif binarised.tif
```

This will create an output folder named `fiber_analysis_output/` containing CSV files and plots.

### Example Command

```bash
python track_sdc_angle.py InputData/skeleton.tif InputData/binarised.tif
```

## Output

- `leaf_to_fixed_node_metric.csv`: Contains metrics for each fixed length point along the fiber including contour length, vertical height, and angle measurements.
- `linear_fitted_slope_vs_lengths.csv`: Summarizes the rate of angle change (slope) at different lengths along the fiber.
- `Linear_fit_of_SDC_Angles.png`: Visual plot showing the relationship between fiber length and angle change rates.

## Output CSV Formats

### `leaf_to_fixed_node_metric.csv`

| Column | Description |
|--------|-------------|
| length | Fixed length along fiber from start node |
| contour_length | Cumulative contour length to the fixed point |
| vertical_heights | Vertical distance from start node to fixed point |
| angles | Angle measurement at the fixed point |

Example:
```csv
length,contour_length,vertical_heights,angles
10,15.24,8.5,143.2
10,16.12,9.1,142.7
...
```

### `linear_fitted_slope_vs_lengths.csv`

| Column | Description |
|--------|-------------|
| length | Fixed length along fiber from start node |
| slopes | Rate of angle change over time |

Example:
```csv
length,slopes
10,-0.034
20,-0.028
...
```

## Analysis Process

The tool performs the following steps:
1. Converts skeleton images to network graphs
2. Identifies key points (leaf nodes and branch points)
3. Measures angles at fixed distances along fibers
4. Tracks angle changes across frames
5. Calculates linear fits to quantify bending rates
6. Generates visualizations and summary data

## Example Plot 

<img src="https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis/blob/main/SDC_Fiber_Bending/fiber_analysis_output/Linear_fit_of_SDC_Angles.png" width="400"/>
