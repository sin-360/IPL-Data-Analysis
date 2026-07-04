# IPL Matches Analysis Project

This project now includes a cleaner exploratory analysis workflow plus a baseline predictive model for predicting whether Team 1 wins an IPL match.

## What has been improved

- Added a structured notebook version for easier navigation and presentation.
- Added a reusable Python script for data preparation, plotting, and modeling.
- Included a simple logistic regression model using team, venue, toss, and toss-win features.
- Added outputs such as summary plots and model metrics.

## Project files

- [IPL_Matches_EDA.ipynb](IPL_Matches_EDA.ipynb) - original exploratory analysis notebook
- [IPL_Matches_EDA_Improved/IPL_Matches_EDA_Improved.ipynb](IPL_Matches_EDA_Improved/IPL_Matches_EDA_Improved.ipynb) - improved notebook
- [IPL_Matches_EDA_Improved/IPL_Matches_EDA_Structured.ipynb](IPL_Matches_EDA_Improved/IPL_Matches_EDA_Structured.ipynb) - clearer structured version
- [IPL_Matches_EDA_Improved/predict_match_outcome.py](IPL_Matches_EDA_Improved/predict_match_outcome.py) - runnable modeling pipeline

## Run the new workflow

1. Open [IPL_Matches_EDA_Improved/IPL_Matches_EDA_Structured.ipynb](IPL_Matches_EDA_Improved/IPL_Matches_EDA_Structured.ipynb) in Jupyter or VS Code.
2. Or run the script from the project root:

```bash
python IPL_Matches_EDA_Improved/predict_match_outcome.py
```

## Dependencies

Install the required packages with:

```bash
pip install -r IPL_Matches_EDA_Improved/requirements.txt
```

## Next steps

- Add recent form or head-to-head features.
- Compare logistic regression with random forest or XGBoost.
- Build a dashboard for interactive match insights.

