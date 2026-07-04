# IPL Matches EDA Improved

This project is an improved version of the IPL data analysis notebook. It uses the same IPL CSV datasets and adds better visualizations, cleaner data preparation, new exploratory analysis sections, and a predictive modeling workflow.

## What is included

- Local data loading from the existing CSV files
- Data cleaning and team name standardization
- Enhanced plots for match counts, winning ratios, toss decisions, venues, and seasons
- New analysis of top batters, wicket takers, venue performance, and toss advantage
- A logistic regression model to predict whether Team 1 wins based on toss result, venue, and teams

## Files

- `IPL_Matches_EDA_Improved.ipynb` — improved notebook with analysis and modeling
- `requirements.txt` — package list for the notebook

## How to use

1. Keep the CSV files in the parent folder `..` relative to the notebook, or update the file paths in the notebook.
2. Install dependencies:

```bash
pip install -r IPL_Matches_EDA_Improved/requirements.txt
```

3. Open `IPL_Matches_EDA_Improved/IPL_Matches_EDA_Improved.ipynb` in Jupyter or VS Code.
4. Run the notebook cells sequentially.

## Notes

- The notebook uses `pandas`, `seaborn`, `matplotlib`, `plotly`, and `scikit-learn`.
- The predictive model is a baseline logistic regression and can be extended further with more features.
