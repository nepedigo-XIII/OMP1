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

def generate_summary(df):
    print("Original DataFrame Summary:")
    print(df.head())
    print()
    print(df.info())

def create_forecast_single(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    if 'Year' not in df.columns or 'Madison Eligibles' not in df.columns:
        raise ValueError("Required columns 'Year' and 'Madison Eligibles' are missing.")

    X = df[['Year']].to_numpy()                 
    y = df['Madison Eligibles'].to_numpy()      

    X_adjusted = X - 2017
    model_adjusted = LinearRegression().fit(X_adjusted, y)
    y_pred_adjusted = model_adjusted.predict(X_adjusted)

    print("\n===========================================================================\n")

    mse = mean_squared_error(y, y_pred_adjusted)
    r2 = r2_score(y, y_pred_adjusted)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Coefficient: {model_adjusted.coef_[0]} (ie. Slope of line, impact of Year on Eligibles)")
    print(f"Intercept: {model_adjusted.intercept_}\n")
    print("\n===========================================================================\n")

    future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
    future_years_adjusted = future_years - 2017
    future_predictions = model_adjusted.predict(future_years_adjusted)

    print("Future Predictions for Number of Eligibles in Madison County:\n")
    for year, pred in zip(future_years.ravel(), future_predictions):
        print(f"Year: {year}, Predicted Eligibles: {pred:.0f}")
        
    plt.figure(figsize=(10, 6))
    plt.scatter(X.ravel(), y, label='Actual Data')
    plt.plot(X.ravel(), y_pred_adjusted, linewidth=2, label='Regression Line')
    plt.scatter(future_years.ravel(), future_predictions, marker='x', s=100, label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('Madison Eligibles')
    plt.title('Madison Eligibles Forecasting (Single Variable)')
    plt.legend()
    plt.grid(True)
    plt.show()

    input("\nPress Enter to continue...")
    plt.close()

    forecast_df = pd.DataFrame({
        'Year': future_years.ravel(),
        'Predicted Madison Eligibles': np.round(future_predictions).astype(int)
    })

    forecast_df.to_csv('madison_forecast_single_var.csv', index=False)
    print("Forecasted data saved to 'madison_forecast_single_var.csv'.")

# No output, only returns a dictionary of forecasts for the same future years
def return_forecast_single(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    if 'Year' not in df.columns or 'Madison Eligibles' not in df.columns:
        raise ValueError("Required columns 'Year' and 'Madison Eligibles' are missing.")
    
    X = df[['Year']].to_numpy()                 
    y = df['Madison Eligibles'].to_numpy()      

    X_adjusted = X - 2017
    model_adjusted = LinearRegression().fit(X_adjusted, y)
    y_pred_adjusted = model_adjusted.predict(X_adjusted)

    mse = mean_squared_error(y, y_pred_adjusted)
    r2 = r2_score(y, y_pred_adjusted)

    future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
    future_years_adjusted = future_years - 2017
    future_predictions = model_adjusted.predict(future_years_adjusted)

    print("Future Predictions for Number of Eligibles in Madison County:\n")
    for year, pred in zip(future_years.ravel(), future_predictions):
        print(f"[Historical Single Var] Year: {year}, Predicted Eligibles: {pred:.0f}")

    forecast_dict = {year: pred for year, pred in zip(future_years.ravel(), future_predictions)}
    return forecast_dict

def create_forecast_multiple(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

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
        print("Running baseline Year-only...\n")
        result = return_forecast_single(df)
        forecasts['year'] = np.array(list(result.values()))

    if vars['unemployment']:
        print("Running Unemployment...\n")
        if 'State Unemployment' not in df.columns:
            raise ValueError("Column 'State Unemployment' is missing.")
        
        X_unemp = df[['Year', 'State Unemployment']].to_numpy()
        y = df['Madison Eligibles'].to_numpy()

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
            print(f"[Unemployment] Year: {year}, Predicted Eligibles: {pred:.0f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_unemp[:, 0] + 2017, y, label='Historical Eligibles')
        plt.plot(X_unemp[:, 0] + 2017, y_pred_unemp, linewidth=2, label='Unemployment Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_unemp, marker='x', s=100, label='Unemployment Based Future Eligibles Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Madison Eligibles')
        plt.title('Madison Eligibles Forecasting (Unemployment Model)')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue...")
        plt.close()

        forecasts['unemployment'] = future_predictions_unemp
    
    if vars['cpi']:
        print("Running Consumer Price Index...\n")
        if 'Consumer Price Index' not in df.columns:
            raise ValueError("Column 'CPI' is missing.")
        
        X_cpi = df[['Year']].to_numpy()
        y_cpi = df['Consumer Price Index'].to_numpy()
        X_cpi_adjusted = X_cpi - 2017
        cpi_model = LinearRegression().fit(X_cpi_adjusted, y_cpi)
        y_cpi_pred = cpi_model.predict(X_cpi_adjusted)

        X_mad_cpi = df[['Consumer Price Index']].to_numpy()
        y_mad_cpi = df['Madison Eligibles'].to_numpy()
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
            print(f"[CPI] Year: {year}, Predicted Eligibles: {pred:.0f}")
        
        # Plot CPI vs eligibles
        plt.figure(figsize=(10, 6))
        plt.scatter(X_mad_cpi.ravel(), y_mad_cpi, label='Historical Eligibles')
        plt.plot(X_mad_cpi.ravel(), y_mad_cpi_pred, linewidth=2, label='CPI Regression Line', color='green')
        plt.scatter(future_X_mad_cpi.ravel(), future_predictions_cpi, marker='x', s=100, label='CPI Future Predictions', color='red')
        plt.xlabel('Consumer Price Index')
        plt.ylabel('Madison Eligibles')
        plt.title('Madison Eligibles vs. CPI (CPI Model)')
        plt.legend()
        plt.grid(True)
        plt.show()
        input("\nPress Enter to continue to final CPI Projection...") 
        plt.close()

        # Plot eligibles over time with CPI predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(X_cpi.ravel(), y_mad_cpi, label='Historical Eligibles')
        plt.plot(X_cpi.ravel(), y_mad_cpi_pred, linewidth=2, label='CPI Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_cpi, marker='x', s=100, label='CPI Based Future Eligibles Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Madison Eligibles')
        plt.title('Madison Eligibles Forecasting Using Predicted Future CPI (CPI Model)')
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
        y_mad_income = df['Madison Eligibles'].to_numpy()
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
            print(f"[Income] Year: {year}, Predicted Eligibles: {pred:.0f}")
        
        # Plot income vs eligibles
        plt.figure(figsize=(10, 6))
        plt.scatter(X_mad_income.ravel(), y_mad_income, label='Actual Data')
        plt.plot(X_mad_income.ravel(), y_mad_income_pred, linewidth=2, label='Income Regression Line', color='green')
        plt.scatter(future_X_mad_income.ravel(), future_predictions_income, marker='x', s=100, label='Income Future Predictions', color='red')
        plt.xlabel('Median Household Income')
        plt.ylabel('Madison Eligibles')
        plt.title('Madison Eligibles vs. Median Household Income (Income Model)')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue to final Income Projection...")
        plt.close()

        # Plot eligibles over time with income predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(X_income.ravel(), y_mad_income, label='Historical Eligibles')
        plt.plot(X_income.ravel(), y_mad_income_pred, linewidth=2, label='Income Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_income, marker='x', s=100, label='Income Based Future Eligibles Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Madison Eligibles')
        plt.title('Madison Eligibles Forecasting Using Predicted Future Median Household Income (Income Model)')
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
        y = df['Madison Eligibles'].to_numpy()

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
            print(f"[SNAP] Year: {year}, Predicted Eligibles: {pred:.0f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(x_snap[:, 0] + 2017, y, label='Actual Data')
        plt.plot(x_snap[:, 0] + 2017, y_pred_snap, linewidth=2, label='SNAP Regression Line', color='green')
        plt.scatter(future_years.ravel(), future_predictions_snap, marker='x', s=100, label='SNAP Based Future Eligibles Predictions', color='red')
        plt.xlabel('Year')
        plt.ylabel('Madison Eligibles')
        plt.title('Madison Eligibles Forecasting (SNAP Model)')
        plt.legend()
        plt.grid(True)
        plt.show()

        input("\nPress Enter to continue...")
        plt.close()

        forecasts['snap'] = future_predictions_snap

    # Final combination of forecasts
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
    y_combined = df['Madison Eligibles'].to_numpy()
    X_combined_adjusted = X_combined - 2017
    combined_model = LinearRegression().fit(X_combined_adjusted, y_combined)
    y_combined_pred = combined_model.predict(X_combined_adjusted)

    print("\n===========================================================================\n")
    print("Combined Future Predictions for Number of Eligibles in Madison County:")

    for year, pred in zip(future_years.ravel(), combined_forecast):
        print(f"[Combined Model] Year: {year}, Predicted Eligibles: {pred:.0f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Year'], df['Madison Eligibles'], label='Actual Data')
    plt.plot(X_combined.ravel(), y_combined_pred, linewidth=2, label='Combined Regression Line', color='orange')
    plt.scatter(future_years.ravel(), combined_forecast, marker='x', s=100, label='Combined Future Predictions', color='purple')
    plt.xlabel('Year')
    plt.ylabel('Madison Eligibles')
    plt.title('Madison Eligibles Forecasting (Multiple Variables Combined)')
    plt.legend()
    plt.grid(True)
    plt.show()

    input("\nPress Enter to continue...")
    plt.close()

    forecast_df = pd.DataFrame({
        'Year': future_years.ravel(),
        'Predicted Madison Eligibles': np.round(combined_forecast).astype(int)
    })

    forecast_df.to_csv('madison_forecast_multiple_var.csv', index=False)
    print("Forecasted data saved to 'madison_forecast_multiple_var.csv'.")

# Main function to run the script
def main():

    clear_screen()
    print("------ Madison County Eligibles Forecasting (Single Variable) ------\n")
    df = toDF()
    print()
    create_forecast_single(df)

    print("\n\nContinue to multiple regression? (y/n): ", end="")
    cont = input().strip().lower()
    if cont == 'y':
        clear_screen()
        print("--- Madison County Eligibles Forecasting (Multiple Variable) ---\n")
        create_forecast_multiple(df)
        print("\nForecasting complete. Exit? (y/n): ", end="")
        exit_input = input().strip().lower()
        if exit_input == 'y':
            print("Exiting program.")
        else:
            print("Restarting program...\n")
            main()
    else:
        print("Exiting program.")


if __name__ == "__main__":
    main()