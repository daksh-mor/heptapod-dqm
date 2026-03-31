# ML4DQM Screenshot Guide

Place your screenshots in this folder with these exact filenames:

## Required Images

1. **eda_tool_screenshot.png** (or .jpg)
   - Screenshot of the EDA tool showing data exploration
   - Used in: README.md → "Explore Your Data" section
   - Suggested: Show histograms and heatmaps

2. **training_tool_screenshot.png** (or .jpg)
   - Screenshot of the training tool in action
   - Used in: README.md → "Train a Model" section
   - Suggested: Show training progress, loss curves, metrics

3. **evaluation_table_screenshot.png** (or .jpg)
   - Screenshot of the evaluation results
   - Used in: README.md → "Evaluate & Find Thresholds" section
   - Suggested: Show ROC curve, metrics table, threshold computation

## Optional Images

4. **anomaly_vs_roc.png** (or .jpg)
   - Plot of anomaly strength vs ROC AUC
   - Can be added if you want to show the relationship
   - Generated using matplotlib

## How to Add Images

Once you have your screenshots:

1. Take screenshots of the Streamlit UI
2. Save them as PNG or JPG in this folder
3. Name them exactly as listed above
4. The README.md will automatically display them

If filenames don't match, the README will show broken image links - just rename and they'll appear.

## Image Tips

- **Resolution**: 1024px width or more (readable on all devices)
- **Format**: PNG (lossless) or JPG (smaller file size)
- **Content**: Show the actual tool output/UI, not terminal windows
- **Clear**: Make sure text and plots are easy to read
