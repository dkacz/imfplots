# IMF Plot Data Extractor

This tool extracts numeric data from plot images (PNG, JPG) using advanced image processing techniques.

## Features

- **Automatic plot type detection**: Detects bar charts, scatter plots, and combined visualizations
- **Color-based series extraction**: Identifies different data series by color
- **Multiple output formats**: Exports data to both CSV and JSON formats
- **Batch processing**: Process all plots in a directory at once
- **High accuracy**: Uses OpenCV and machine learning for precise data extraction

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Process all plots in the current directory

```bash
python extract_plot_data.py
```

This will:
- Scan the current directory for image files (.png, .jpg, .jpeg)
- Extract data from each plot
- Save results to the `extracted_data/` directory

### Process plots from a specific directory

```bash
python extract_plot_data.py --input /path/to/plots --output /path/to/output
```

### Process a single plot

```bash
python extract_plot_data.py --single plot_image.jpg --output results/
```

### Specify plot type

```bash
python extract_plot_data.py --plot-type bar    # For bar charts
python extract_plot_data.py --plot-type scatter # For scatter plots
python extract_plot_data.py --plot-type auto    # Auto-detect (default)
```

## Output Format

### CSV Format

For **bar charts**:
```csv
series,x_position,value,color_r,color_g,color_b
series_0,0,85.2,100,149,237
series_0,1,82.1,100,149,237
series_1,0,60.3,220,20,60
```

For **scatter plots**:
```csv
series,x_value,y_value,color_r,color_g,color_b
series_0,25.5,68.3,100,149,237
series_0,30.2,72.1,100,149,237
series_1,15.8,45.2,220,20,60
```

### JSON Format

```json
{
  "plot_type": "bar",
  "image_path": "plot.jpg",
  "plot_area": [50, 40, 450, 380],
  "data": {
    "series_0": {
      "x_positions": [0, 1, 2],
      "values": [85.2, 82.1, 78.5],
      "color": [100, 149, 237]
    }
  }
}
```

## How It Works

1. **Plot Area Detection**: Identifies the main plotting region using edge detection
2. **Color Extraction**: Uses K-means clustering to identify dominant colors representing different data series
3. **Data Extraction**:
   - For bar charts: Measures bar heights and positions
   - For scatter plots: Detects point coordinates
4. **Normalization**: Converts pixel coordinates to normalized values (0-100 scale)
5. **Export**: Saves data in CSV and JSON formats

## Plot Types Supported

- Vertical bar charts (single or grouped)
- Horizontal bar charts
- Scatter plots
- Line plots with markers
- Combined bar and line plots

## Notes

- Values are extracted as percentages of the plot area (0-100 scale)
- Colors are represented in RGB format
- The tool works best with clear, high-resolution plot images
- For axis labels and exact values, manual calibration may be needed

## Example Workflow

```bash
# 1. Extract data from all plots
python extract_plot_data.py

# 2. Check the extraction summary
cat extracted_data/extraction_summary.json

# 3. Load data in Python for analysis
import pandas as pd
data = pd.read_csv('extracted_data/plot_name.csv')
print(data)
```

## Troubleshooting

**Issue**: Colors not detected correctly
- **Solution**: The tool uses automatic color detection. If results are inaccurate, try adjusting the `color_tolerance` parameter in the code.

**Issue**: Plot area not detected correctly
- **Solution**: The tool uses heuristic boundaries. For non-standard plots, you may need to manually adjust the plot area detection logic.

**Issue**: Low accuracy for complex plots
- **Solution**: Ensure images are high resolution and have good contrast. Complex overlapping plots may require manual adjustment.

## Advanced Usage

You can also use the `PlotDataExtractor` class directly in Python:

```python
from extract_plot_data import PlotDataExtractor

# Load and extract
extractor = PlotDataExtractor('my_plot.jpg')
data = extractor.extract_data(plot_type='bar')

# Save results
extractor.save_to_csv(data, 'output.csv')
extractor.save_to_json(data, 'output.json')

# Access raw data
print(data['data'])
```

## License

This tool is provided as-is for extracting data from IMF economic plots.
