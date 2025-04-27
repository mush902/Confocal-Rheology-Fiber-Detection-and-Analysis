# Agarose Connectivity Junction Density Analysis

A Python-based tool for analyzing fiber skeleton images to detect junction points based on graph connectivity and quantify the network's degree distribution.

## Features

- Constructs a graph from binary skeleton images
- Detects and removes spurious diagonal connections (cycle detection)
- Identifies junction nodes with degree ≥ 3
- Calculates degree distribution across the network
- Visualizes junction nodes overlaid on confocal fiber images

## Dependencies

Dependencies can be found in the `requirements.txt` file in the root directory.

## Installation

```bash
git clone https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis.git
cd Confocal-Rheology-Fiber-Detection-and-Analysis/
pip install -r requirements.txt
```

## Usage

### Input Files Required

- `skeleton.tif`: Binary skeleton image sequence (TIFF stack)
- `confocal.tif`: Original confocal fiber image sequence (TIFF stack) for overlay visualization

### Running the Analysis

```bash
python use_graph_junction.py skeleton.tif confocal.tif
```

This will create an output folder named `junction_analysis_output/` containing CSV files and plots.

### Example Command

```bash
python3 use_graph_junction.py path/to/skeleton.tif path/to/confocal.tif
```

## Output

- `junction_nodes.csv`: Contains (X, Y) coordinates and degree of each junction node detected.
- `degree_distribution.csv`: Summarizes the number of nodes for each degree along with the total image area.
- Visual plots showing detected junction points overlaid on the original confocal images.

## Output CSV Formats

### `junction_nodes.csv`
| Column | Description |
|--------|-------------|
| X      | X-coordinate of junction node |
| Y      | Y-coordinate of junction node |
| Degree | Degree (number of connections) at the node |

Example:
```csv
X,Y,Degree
120,200,3
230,145,4
...
```

### `degree_distribution.csv`
| Column | Description |
|--------|-------------|
| Degree | Node degree |
| Total Pixel Count | Number of nodes with this degree |
| Total Area | Total image pixel area (constant for the image) |

Example:
```csv
Degree,Total Pixel Count,Total Area
3,120,1048576
4,45,1048576
5,8,1048576
```

## Plots

The script also generates plots showing:

- The original confocal fiber image
- The overlaid skeleton graph
- Junction nodes highlighted with different colors:
  - Red: Degree = 3
  - Green: Degree = 4
  - Blue: Degree ≥ 5

## Example

<img src="https://github.com/mush902/Confocal-Rheology-Fiber-Detection-and-Analysis/blob/main/Agarose_Connectivity_Junction_Density/Graph-Conc%20-%200.25/overlay.png?raw=true" width="400"/>


