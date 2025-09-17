import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clear_screen():
    if os.name == 'nt':
        os.system('cls')  # Command for Windows
    else:
        os.system('clear') # Command for Linux/macOS

def toDF():
    try:
        df = pd.read_excel('madisonData.xlsx')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: The file 'madisonData.xlsx' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()
    
    return df


def reportEligiblesBenefit(df):
    print("==============================================================\n")
    print("\tPart 2: Data Collection and Cleaning\n")
    print("==============================================================\n")
    print("\t> Madison County Eligibles <\t\n\n")

    # Print Year and Eligibles 
    for index, row in df.iterrows():
        print(f"Year: {row['Year']:.0f}\t|\t Madison Eligibles: {row['Madison Eligibles']:,.0f}")

    print("\n==============================================================\n")
    print("\t> Madison County Benefit Cost <\t\n")

    for index, row in df.iterrows():
        print(f"Year: {row['Year']:.0f}\t|\t Madison Benefit Cost: ${row['Benefit Payments']:,.0f}")

    print("==============================================================\n")



def create_forecast_single(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    print("====================================================================\n")
    print("\t      Regression Based on the Data Only")

    # Check for required columns
    if 'Year' not in df.columns or 'Benefit Payments' not in df.columns:
        raise ValueError("Required columns 'Year' and 'Benefit Payments' are missing.")

    # Optionally filter to start at 2017
    df = df[df['Year'] >= 2017].copy()

    # Features and target (represent Benefit Payments in $1,000s)
    X = df[['Year']]
    y = df[['Benefit Payments']] / 1_000  # in $1,000s

    # Fit linear regression directly (no scaling)
    model = LinearRegression()
    model.fit(X, y)

    # Predictions on training data
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0][0]       # coefficient for "Year"
    intercept = model.intercept_[0] # intercept

    print("\n====================================================================\n")
    print(f"Intercept: ${df.loc[df['Year'] == 2017, 'Benefit Payments'].iloc[0]:,.0f}")
    print(f"Slope (Coefficient): {slope:,.0f} $ / yr")
    print(f"Mean Squared Error: {mse:,.0f}")
    print(f"R^2 Score: {r2:.4f}")

    # Forecast 2023-2027
    future_years = pd.DataFrame({'Year': [2023, 2024, 2025, 2026, 2027]})
    future_predictions = model.predict(future_years)
    future_years['Predicted Benefit Payments'] = future_predictions

    # Print future predictions
    print("\n====================================================================\n")
    print("\tFuture Year Predictions:\n")
    for _, row in future_years.iterrows():
        print(f"Year: {int(row['Year'])}\t|\t Predicted Benefit Cost: ${row['Predicted Benefit Payments']:,.2f}k")
    print("\n====================================================================\n")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Year'], y, color='blue', label='Actual Data')
    plt.plot(df['Year'], y_pred, color='red', linewidth=2, label='Regression Line')
    plt.scatter(future_years['Year'], future_years['Predicted Benefit Payments'],
                color='green', marker='x', s=100, label='Future Predictions')

    plt.title('Madison County Benefit Cost Forecasting (Linear Regression)')
    plt.xlabel('Year')
    plt.ylabel('Benefit Payments in $1,000s (USD)')

    # Format y-axis: no scientific notation, add commas
    plt.ticklabel_format(style='plain', axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))

    plt.legend()
    plt.grid(True)
    plt.show()

    forecast_df = pd.DataFrame({
    'Year': future_years['Year'],
    'Predicted Benefit Payments': future_predictions.ravel()
    })

    forecast_df.to_csv('madison_forecast_single_var.csv', index=False)
    print("Forecasted data saved to 'madison_forecast_single_var.csv'.")
    


    
def create_forecast_multiple(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    vars = {
        'year': True,
        'unemployment': True,
        'cpi': True,
        'income': True,
        'snap': True,  
    }

    weights = {
        'year': 0.15,
        'unemployment': 0.2,
        'cpi': 0.2,
        'income': 0.15,  
        'snap': 0.3,    
    }

    forecasts = {}

    if vars['year']:
    
        # 1. Read CSV
        csv_file = 'madison_forecast_single_Var.csv'
        df = pd.read_csv(csv_file)
        
        # 2. Prepare data
        X = df[["Year"]]  # Feature
        y = df[[target_col]]  # Target
        
        # 3. Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # 4. Generate future years
        future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
        future_preds = model.predict(future_years)
        
        # 5. Create future dataframe
        df_future = pd.DataFrame({
            "Year": future_years.flatten(),
            target_col: future_preds.flatten()
        })
        
        # 6. Append to original
        df_extended = pd.concat([df, df_future], ignore_index=True)
        
        # 7. Save updated CSV
        df_extended.to_csv(output_file, index=False)
 

    if vars['unemployment']:
        print("Running Unemployment...\n")
        if 'State Unemployment' not in df.columns:
            raise ValueError("Column 'State Unemployment' is missing.")
        
        X_unemp = df[['Year', 'State Unemployment']].to_numpy()
        y = df['Benefit Payments'].to_numpy()

        X_unemp[:, 0] = X_unemp[:, 0] - 2017

        model_unemp = LinearRegression().fit(X_unemp, y)
        y_pred_unemp = model_unemp.predict(X_unemp)

        mse_unemp = mean_squared_error(y, y_pred_unemp)
        r2_unemp = r2_score(y, y_pred_unemp)

        print("\n===========================================================================\n")
        print(f"[Unemployment] Mean Squared Error: {mse_unemp}")
        print(f"[Unemployment] R^2 Score: {r2_unemp}")
        print(f"[Unemployment] Coefficients: {model_unemp.coef_}")
        print(f"[Unemployment] Intercept: {model_unemp.intercept_}\n")

        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        last_unemp_rate = df['State Unemployment'].iloc[-1]
        future_unemp_rates = np.full(future_years.shape, last_unemp_rate)
        future_X_unemp = np.hstack((future_years - 2017, future_unemp_rates))
        future_predictions_unemp = model_unemp.predict(future_X_unemp)

        for year, pred in zip(future_years.ravel(), future_predictions_unemp):
            print(f"[Unemployment] Year: {year}, Predicted Benefit Payments: {pred:.0f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_unemp[:, 0] + 2017, y, label='Historical Benefit Payments')
        plt.plot(X_unemp[:, 0] + 2017, y_pred_unemp, linewidth=2, label='Unemployment Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_unemp, marker='x', s=100, label='Future Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Benefit Payments')
        plt.title('Benefit Payments Forecasting (Unemployment Model)')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue...")
        plt.close()

        forecasts['unemployment'] = future_predictions_unemp
    
    if vars['cpi']:
        print("Running Consumer Price Index...\n")
        if 'Consumer Price Index' not in df.columns:
            raise ValueError("Column 'Consumer Price Index' is missing.")
        
        X_cpi = df[['Year']].to_numpy()
        y_cpi = df['Consumer Price Index'].to_numpy()
        X_cpi_adjusted = X_cpi - 2017
        cpi_model = LinearRegression().fit(X_cpi_adjusted, y_cpi)
        y_cpi_pred = cpi_model.predict(X_cpi_adjusted)

        X_mad_cpi = df[['Consumer Price Index']].to_numpy()
        y_mad_cpi = df['Benefit Payments'].to_numpy()
        cpi_mad_model = LinearRegression().fit(X_mad_cpi, y_mad_cpi)
        y_mad_cpi_pred = cpi_mad_model.predict(X_mad_cpi)

        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        future_cpi = cpi_model.predict(future_years - 2017)
        future_X_mad_cpi = future_cpi.reshape(-1, 1)
        future_predictions_cpi = cpi_mad_model.predict(future_X_mad_cpi)
        mse_cpi = mean_squared_error(y_mad_cpi, y_mad_cpi_pred)
        r2_cpi = r2_score(y_mad_cpi, y_mad_cpi_pred)

        print("\n===========================================================================\n")
        print(f"[CPI] Mean Squared Error: {mse_cpi}")
        print(f"[CPI] R^2 Score: {r2_cpi}")
        print(f"[CPI] Coefficients: {cpi_mad_model.coef_}")
        print(f"[CPI] Intercept: {cpi_mad_model.intercept_}\n")

        for year, pred in zip(future_years.ravel(), future_predictions_cpi):
            print(f"[CPI] Year: {year}, Predicted Benefit Payments: {pred:.0f}")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_mad_cpi.ravel(), y_mad_cpi, label='Historical Benefit Payments')
        plt.plot(X_mad_cpi.ravel(), y_mad_cpi_pred, linewidth=2, label='CPI Regression Line', color='green')
        plt.scatter(future_X_mad_cpi.ravel(), future_predictions_cpi, marker='x', s=100, label='Future Predictions', color='red')
        plt.xlabel('Consumer Price Index')
        plt.ylabel('Benefit Payments')
        plt.title('Benefit Payments vs. CPI')
        plt.legend()
        plt.grid(True)
        plt.show()
        input("\nPress Enter to continue to final CPI Projection...") 
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(X_cpi.ravel(), y_mad_cpi, label='Historical Benefit Payments')
        plt.plot(X_cpi.ravel(), y_mad_cpi_pred, linewidth=2, label='CPI Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_cpi, marker='x', s=100, label='Future Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Benefit Payments')
        plt.title('Benefit Payments Forecasting Using Predicted Future CPI')
        plt.legend()
        plt.grid(True)
        plt.show()
        input("\nPress Enter to continue...")
        plt.close()

        forecasts['cpi'] = future_predictions_cpi

    if vars['income']:
        if 'Median Household Income' not in df.columns:
            raise ValueError("Column 'Median Household Income' is missing.")
        
        X_income = df[['Year']].to_numpy()
        y_income = df['Median Household Income'].to_numpy()

        X_income_adjusted = X_income - 2017
        income_model = LinearRegression().fit(X_income_adjusted, y_income)

        y_income_pred = income_model.predict(X_income_adjusted)

        X_mad_income = df[['Median Household Income']].to_numpy()
        y_mad_income = df['Benefit Payments'].to_numpy()
        income_mad_model = LinearRegression().fit(X_mad_income, y_mad_income)
        y_mad_income_pred = income_mad_model.predict(X_mad_income)

        mse_income = mean_squared_error(y_mad_income, y_mad_income_pred)
        r2_income = r2_score(y_mad_income, y_mad_income_pred)

        print("\n===========================================================================\n")
        print(f"[Income] Mean Squared Error: {mse_income}")
        print(f"[Income] R^2 Score: {r2_income}")
        print(f"[Income] Coefficients: {income_mad_model.coef_}")
        print(f"[Income] Intercept: {income_mad_model.intercept_}\n")

        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        future_income = income_model.predict(future_years - 2017)
        future_X_mad_income = future_income.reshape(-1, 1)
        future_predictions_income = income_mad_model.predict(future_X_mad_income)

        for year, pred in zip(future_years.ravel(), future_predictions_income):
            print(f"[Income] Year: {year}, Predicted Benefit Payments: {pred:.0f}")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_mad_income.ravel(), y_mad_income, label='Actual Data')
        plt.plot(X_mad_income.ravel(), y_mad_income_pred, linewidth=2, label='Income Regression Line', color='green')
        plt.scatter(future_X_mad_income.ravel(), future_predictions_income, marker='x', s=100, label='Future Predictions', color='red')
        plt.xlabel('Median Household Income')
        plt.ylabel('Benefit Payments')
        plt.title('Benefit Payments vs. Median Household Income')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue to final Income Projection...")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(X_income.ravel(), y_mad_income, label='Historical Benefit Payments')
        plt.plot(X_income.ravel(), y_mad_income_pred, linewidth=2, label='Income Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_income, marker='x', s=100, label='Future Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Benefit Payments')
        plt.title('Benefit Payments Forecasting Using Predicted Future Income')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue...")
        plt.close()

        forecasts['income'] = future_predictions_income

    if vars['snap']:
        if 'SNAP Recipients' not in df.columns:
            raise ValueError("Column 'SNAP Recipients' is missing.")
        
        x_snap = df[['Year', 'SNAP Recipients']].to_numpy()
        y = df['Benefit Payments'].to_numpy()

        x_snap[:, 0] = x_snap[:, 0] - 2017

        model_snap = LinearRegression().fit(x_snap, y)
        y_pred_snap = model_snap.predict(x_snap)

        mse_snap = mean_squared_error(y, y_pred_snap)
        r2_snap = r2_score(y, y_pred_snap)

        print("\n===========================================================================\n")
        print(f"[SNAP] Mean Squared Error: {mse_snap}")
        print(f"[SNAP] R^2 Score: {r2_snap}")
        print(f"[SNAP] Coefficients: {model_snap.coef_}")
        print(f"[SNAP] Intercept: {model_snap.intercept_}\n")

        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        last_snap = df['SNAP Recipients'].iloc[-1]
        future_snaps = np.full(future_years.shape, last_snap)
        future_x_snap = np.hstack((future_years - 2017, future_snaps))
        future_predictions_snap = model_snap.predict(future_x_snap)

        for year, pred in zip(future_years.ravel(), future_predictions_snap):
            print(f"[SNAP] Year: {year}, Predicted Benefit Payments: {pred:.0f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(x_snap[:, 0] + 2017, y, label='Actual Data')
        plt.plot(x_snap[:, 0] + 2017, y_pred_snap, linewidth=2, label='SNAP Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_snap, marker='x', s=100, label='Future Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Benefit Payments')
        plt.title('Benefit Payments Forecasting (SNAP Model)')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue...")
        plt.close()

        forecasts['snap'] = future_predictions_snap

    # Final combination
    print("\n===========================================================================\n")
    print("Combining forecasts from selected variables...\n")

    combined_forecast = np.zeros(future_years.shape[0])
    total_weight = 0
    for var, pred in forecasts.items():
        weight = weights.get(var, 0)
        print(f"Adding {var} forecast with weight {weight}.")
        combined_forecast += weight * pred
        total_weight += weight
    if total_weight > 0:
        print(f"Total weight: {total_weight}. Normalizing combined forecast.")
        combined_forecast /= total_weight
    else:
        print("No variables selected for forecasting.")
        return
    
    X_combined = df[['Year']].to_numpy()
    y_combined = df['Benefit Payments'].to_numpy()
    X_combined_adjusted = X_combined - 2017
    combined_model = LinearRegression().fit(X_combined_adjusted, y_combined)
    y_combined_pred = combined_model.predict(X_combined_adjusted)

    print("\n===========================================================================\n")
    print("Combined Future Predictions for Benefit Payments:")

    for year, pred in zip(future_years.ravel(), combined_forecast):
        print(f"[Combined Model] Year: {year}, Predicted Benefit Payments: {pred:.0f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Year'], df['Benefit Payments'], label='Actual Data')
    plt.plot(X_combined.ravel(), y_combined_pred, linewidth=2, label='Combined Regression Line', color='orange')
    plt.scatter(future_years.ravel(), combined_forecast, marker='x', s=100, label='Combined Future Predictions', color='purple')
    plt.xlabel('Year')
    plt.ylabel('Benefit Payments')
    plt.title('Benefit Payments Forecasting (Multiple Variables Combined)')
    plt.legend()
    plt.grid(True)
    plt.show()

    input("\nPress Enter to continue...")
    plt.close()

    forecast_df = pd.DataFrame({
        'Year': future_years.ravel(),
        'Predicted Benefit Payments': np.round(combined_forecast).astype(int)
    })

    forecast_df.to_csv('benefit_payments_forecast_multiple_var.csv', index=False)
    print("Forecasted data saved to 'benefit_payments_forecast_multiple_var.csv'.")

    

# Main function to run the script
def main():

    clear_screen()
    df = toDF()

    reportEligiblesBenefit(df)

    print("Continue to Linear Regression Prediction of Benefit Cost? (y/n): ", end="")
    cont = input().strip().lower()

    if cont != 'y' and cont != 'yes' and cont != 'Y' and cont != 'YES':
        print("Exiting program.")
        return

    clear_screen()

    create_forecast_single(df)

    print("\n\nContinue to multiple regression? (y/n): ", end="")
    cont = input().strip().lower()

    if cont != 'y' or cont != 'yes' or cont != 'Y' or cont != 'YES':
        print("Exiting program.")
        return

    

if __name__ == "__main__":
    main()