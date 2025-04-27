# SDC Fiber Diameter Analysis
A Python-based tool for analyzing fiber diameters in skeletal networks by measuring perpendicular cross-sections at midpoints.

## Features
- Constructs graph representations from binary skeleton images
- Detects and eliminates spurious diagonal connections through cycle detection
- Identifies fiber midpoints along each segment
- Calculates perpendicular measurements across fibers by:
  - Finding the midpoint of each fiber segment
  - Computing the perpendicular direction vector
  - Extending measurement lines until reaching white pixels (fiber boundary)
- Generates detailed diameter statistics:
  - Per-component diameter measurements
  - Per-graph average diameters
  - Overall average diameter across the network
- Visualizes results with overlaid measurements on fiber images

## Dependencies
Dependencies can be found in the `requirements.txt` file in the root directory.

## Installation
```bash
git clone https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis.git
cd Confocal-Rheology-Fiber-Detection-and-Analysis/Diameter_Analysis/
pip install -r requirements.txt
```

## Usage
### Input Files Required
- `skeleton.tif`: Binary skeleton image sequence (TIFF stack)
- `binarised.tif`: Binary fiber image sequence (TIFF stack) for boundary detection

### Running the Analysis
```bash
python find_fiber_diameter.py skeleton.tif binarised.tif [component_threshold_length]
```

### Optional Parameters
- `component_threshold_length`: Minimum fiber segment length to include in analysis (default: 10)

### Example Command
```bash
python find_fiber_diameter.py EnhancedContrast/skeleton.tif EnhancedContrast/binarised.tif 15
```

## Output
The script creates an output folder named `fiber_analysis_output/` containing:
- `component_diameters.csv`: Detailed diameter measurements and statistics
- Interactive visualizations showing:
  - Fiber skeleton graph representation
  - Midpoint identification (cyan dots)
  - Perpendicular diameter measurements (cyan lines)

## Output CSV Format
### `component_diameters.csv`
The CSV file contains three sections separated by empty rows:

#### 1. Individual Component Measurements
| Column | Description |
|--------|-------------|
| graph_id | Identifier for the connected graph component |
| component_id | Identifier for the specific fiber segment within its graph |
| diameter | Measured diameter in pixels at the midpoint of the segment |

Example:
```csv
graph_id,component_id,diameter
0,0,12.45
0,1,10.23
1,0,8.76
...
```

#### 2. Per-Graph Statistics
| Column | Description |
|--------|-------------|
| graph_id | Identifier for the connected graph component |
| average_diameter | Mean diameter of all segments in this graph (to 4 decimal places) |
| total_length | Sum of contour lengths of all segments in the graph |

Example:
```csv
graph_id,average_diameter,total_length
0,11.3400,425.6
1,9.7800,320.2
...
```

#### 3. Overall Statistics
| Column | Description |
|--------|-------------|
| overall_average_diameter | Mean diameter across all fiber segments in all graphs (to 4 decimal places) |

Example:
```csv
overall_average_diameter,10.5600
```


## Example Overlay Plot
<img src="https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis/blob/main/Diameter_Analysis/fiber_analysis_output/diameter_overlay.png" width="400"/>


## Directory Structure
```
Diameter_Analysis/
├── find_fiber_diameter.py
├── EnhancedContrast/
│   ├── skeleton.tif
│   ├── binarised.tif
│   ├── colored.tif
│   ├── blurred_colored.tif
│   └── overlay.png
└── fiber_analysis_output/
    └── component_diameters.csv
```
