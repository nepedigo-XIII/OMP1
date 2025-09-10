# OMP1
Project 1: Python Multiple Regression Model

**REQUIRED LIBRARIES**:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**FILES**:
- `RegressionModel.py`: Main script to run the regression analysis and forecasting.
- `madison_data.csv`: Sample dataset containing historical data for Madison, WI.
- `README.md`: This file.

**USAGE**:
1. Ensure you have the required libraries installed.
2. Place `madison_data.csv` in the same directory as `RegressionModel.py`.
3. Run the script using Python:
   ```bash
   python RegressionModel.py
   ```

**DESCRIPTION**:
This project performs multiple linear regression analysis on historical data from Madison, WI, to forecast future values of a target variable (e.g., number of eligibles) based on several predictor variables (e.g., unemployment rate, median income, population). The script includes data loading, preprocessing, model training, evaluation, and visualization of results.