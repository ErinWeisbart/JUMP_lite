#!/usr/bin/env python3
import pandas as pd
import panel as pn
import holoviews as hv
from pathlib import Path

pn.extension()
hv.extension('bokeh')

# Load CSV
csv_path = Path("output/segmentation_comparison_with_mapping/segmentation_comparison_with_mapping_large_cell_count_diff.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} samples")

# Simple scatter plot - no overlay, just one plot
scatter = hv.Scatter(
    df.head(100),
    kdims=['n_true'],
    vdims=['cell_count_diff']
).opts(
    width=600,
    height=600,
    color='blue',
    size=5,
    tools=['hover', 'tap'],
    title='Cell Count Differences (Simple Test)'
)

app = pn.Column(
    pn.pane.Markdown("## Simple Test Viewer"),
    pn.pane.Markdown(f"Loaded {len(df)} samples (showing first 100)"),
    pn.pane.HoloViews(scatter),
    pn.pane.Markdown("If you see the scatter plot above, rendering works!")
)

app.servable()

if __name__ == "__main__":
    pn.serve(app, port=5009, show=True, title="Simple Viewer Test")
