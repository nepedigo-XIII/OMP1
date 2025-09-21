import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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


def eligibles_vs_payment(df):
    print("==============================================================")
    print(" Part 3: Linear Regression of Eligibles vs Benefit Payments")
    print("==============================================================\n")

    # --- Check required columns ---
    required_cols = ['Year', 'Madison Eligibles', 'Benefit Payments']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing.")

    # --- Filter data for plotting ---
    df_plot = df[(df['Year'] >= 2010) & (df['Year'] <= 2022)].copy()

    # --- Features and target (Benefit Payments in $1,000s) ---
    X = df_plot[['Madison Eligibles']].values
    y = df_plot[['Benefit Payments']].values / 1_000  # scale to $1,000s

    # --- Fit linear regression (do not force intercept) ---
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    intercept = model.intercept_[0]  # keep the intercept as computed
    slope = model.coef_[0][0]

    # --- Predictions ---
    y_pred = slope * X + intercept

    # --- Metrics ---
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\n====================================================================\n")
    print(f"Intercept: ${intercept:,.2f}k")
    print(f"Slope (Coefficient): {slope:,.2f} $k per eligible")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R^2 Score: {r2:.4f}")

    # --- Forecast future Benefit Payments based on projected Eligibles ---
    forecast_years = [2023, 2024, 2025, 2026, 2027]
    last_eligible = df_plot['Madison Eligibles'].iloc[-1]
    eligible_growth = (df_plot['Madison Eligibles'].iloc[-1] - df_plot['Madison Eligibles'].iloc[0]) / \
                      (df_plot['Year'].iloc[-1] - df_plot['Year'].iloc[0])

    future_df = pd.DataFrame({'Year': forecast_years})
    future_df['Madison Eligibles'] = [last_eligible + eligible_growth * (i + 1) for i in range(len(forecast_years))]
    future_df['Predicted Benefit Payments'] = slope * future_df['Madison Eligibles'] + intercept

    # --- Print forecast ---
    print("\n====================================================================\n")
    print("\tFuture Year Predictions Based on Eligibles:\n")
    for _, row in future_df.iterrows():
        print(f"Year: {int(row['Year'])}\t|\tPredicted Benefit Cost: ${row['Predicted Benefit Payments']:,.2f}k")
    print("\n====================================================================\n")

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['Madison Eligibles'], y, color='blue', label='Actual Data')
    plt.plot(df_plot['Madison Eligibles'], y_pred, color='red', linewidth=2, label='Regression Line')

    plt.title('Benefit Payments vs Madison Eligibles')
    plt.xlabel('Number of Eligibles')
    plt.ylabel('Benefit Payments in $1,000s (USD)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}".format(int(x))))

    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Save forecast ---
    forecast_save = future_df[['Year', 'Predicted Benefit Payments']].copy()
    forecast_save['Predicted Benefit Payments'] = (forecast_save['Predicted Benefit Payments'] * 1000).round(0)
    forecast_save.to_csv('madison_forecast_eligibles.csv', index=False)
    print("Forecasted data saved to 'madison_forecast_eligibles.csv'.")
    
    return future_df


def create_forecast_single(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    required_cols = ['Year', 'Benefit Payments']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing.")

    # --- Filter data for 2010–2022 ---
    df_plot = df[(df['Year'] >= 2010) & (df['Year'] <= 2022)].copy()

    # --- Features and target (Benefit Payments in $1,000s) ---
    X = df_plot[['Year']].values
    y = df_plot[['Benefit Payments']].values / 1_000  # scale to $1,000s

    # --- Fit linear regression ---
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # --- Use the intercept as computed (do not force non-negative) ---
    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    # --- Predictions ---
    y_pred = slope * X + intercept

    # --- Metrics ---
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\n====================================================================\n")
    print(f"Intercept: ${intercept:,.2f}k")
    print(f"Slope (Coefficient): {slope:,.2f} $k per year")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R^2 Score: {r2:.4f}")

    # --- Forecast future years 2023–2027 ---
    future_years = pd.DataFrame({'Year': [2023, 2024, 2025, 2026, 2027]})
    future_years['Predicted Benefit Payments'] = slope * future_years['Year'] + intercept

    # --- Print future predictions ---
    print("\n====================================================================\n")
    print("\tFuture Year Predictions:\n")
    for _, row in future_years.iterrows():
        print(f"Year: {int(row['Year'])}\t|\t Predicted Benefit Cost: ${row['Predicted Benefit Payments']:,.2f}k")
    print("\n====================================================================\n")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['Year'], y, color='blue', label='Actual Data')
    plt.plot(df_plot['Year'], y_pred, color='red', linewidth=2, label='Regression Line')

    plt.title('Benefit Payments vs Year (2010–2022) with Forecast to 2027')
    plt.xlabel('Year')
    plt.ylabel('Benefit Payments in $1,000s (USD)')

    plt.ticklabel_format(style='plain', axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}".format(int(x))))

    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Save forecast ---
    forecast_df = future_years[['Year', 'Predicted Benefit Payments']].copy()
    forecast_df['Predicted Benefit Payments'] = (forecast_df['Predicted Benefit Payments'] * 1000).round(0)
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
        'year': 0.4,
        'unemployment': 0.2,
        'cpi': 0.35,
        'income': 0.05,  
        'snap': 0.0,    
    }

    forecasts = {}

    if vars.get('year'):
        create_forecast_single(df)


        linearForecast = pd.read_csv("madison_forecast_single_var.csv")
        forecasts['year'] = linearForecast['Predicted Benefit Payments']

        input("\nPress Enter to continue...")

    

    if vars['unemployment']:
        print("Running Unemployment...\n")

        if 'State Unemployment' not in df.columns:
            raise ValueError("Column 'State Unemployment' is missing.")

        # --- Filter data starting from 2010 ---
        df_plot = df[df['Year'] >= 2010].copy()

        # --- Fit linear model: Unemployment Rate vs Year ---
        X_year = df_plot[['Year']].to_numpy()
        y_unemp = df_plot['State Unemployment'].to_numpy()
        unemp_year_model = LinearRegression().fit(X_year, y_unemp)

        # Predict historical unemployment for comparison
        y_unemp_pred = unemp_year_model.predict(X_year)

        # --- Fit model: Benefit Payments vs Unemployment ---
        X_unemp = df_plot[['State Unemployment']].to_numpy()
        y_benefits = df_plot['Benefit Payments'].to_numpy()
        model_unemp = LinearRegression().fit(X_unemp, y_benefits)
        y_benefits_pred = model_unemp.predict(X_unemp)

        # --- Metrics ---
        mse_unemp = mean_squared_error(y_benefits, y_benefits_pred)
        r2_unemp = r2_score(y_benefits, y_benefits_pred)

        print("\n===========================================================================\n")
        print(f"[Unemployment] Mean Squared Error: {mse_unemp:,.2f}")
        print(f"[Unemployment] R^2 Score: {r2_unemp:.4f}")
        print(f"[Unemployment] Coefficients: {model_unemp.coef_[0]:,.2f} $ / Percent")
        print(f"[Unemployment] Intercept: ${model_unemp.intercept_:,.0f}\n")

        # --- Forecast future unemployment for next 5 years ---
        last_year = df_plot['Year'].max()
        future_years = np.arange(last_year + 1, last_year + 6).reshape(-1, 1)
        future_unemp_rates = unemp_year_model.predict(future_years)

        # --- Forecast future benefit payments based on forecasted unemployment ---
        future_predictions_unemp = model_unemp.predict(future_unemp_rates.reshape(-1, 1))

        # --- Plot historical relationship ---
        plt.figure(figsize=(10, 6))
        plt.scatter(X_unemp, y_benefits/1000, label='Historical Benefit Payments', color='blue')
        plt.plot(X_unemp, y_benefits_pred/1000, linewidth=2, label='Unemployment Regression Line', color='green')
        plt.xlabel('Unemployment Rate (%)')
        plt.ylabel('Benefit Payments in $1,000s')
        plt.title('Benefit Payments vs. Unemployment Rate')
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Print future forecasts ---
        print("\nFuture Benefit Payment Forecasts Based on Unemployment:\n")
        for year, pred in zip(future_years.ravel(), future_predictions_unemp):
            print(f"[Unemployment] Year: {year}, Predicted Benefit Payments: ${pred:,.0f}")

        # --- Store in forecasts dictionary ---
        forecasts['unemployment'] = future_predictions_unemp

        input("\nPress Enter to continue...")
        plt.close()


    
    if vars['cpi']:
        print("Running Consumer Price Index...\n")

        # --- Step 1: Year → CPI model ---
        if 'Consumer Price Index' not in df.columns:
            raise ValueError("Column 'Consumer Price Index' is missing.")

        X_year = df[['Year']].to_numpy()
        y_cpi = df['Consumer Price Index'].to_numpy()

        # Center years around 2010 to stabilize intercept
        X_year_adjusted = X_year - 2010
        cpi_model = LinearRegression().fit(X_year_adjusted, y_cpi)
        y_cpi_pred = cpi_model.predict(X_year_adjusted)

        # --- Step 2: CPI → Benefit Payments model ---
        X_cpi = df[['Consumer Price Index']].to_numpy()

        # Center CPI around its mean to prevent large intercepts
        cpi_mean = X_cpi.mean()
        X_cpi_centered = X_cpi - cpi_mean
        y_benefits = df['Benefit Payments'].to_numpy()

        cpi_benefits_model = LinearRegression().fit(X_cpi_centered, y_benefits)
        y_benefits_pred = cpi_benefits_model.predict(X_cpi_centered)

        # --- Forecast future CPI using Step 1 ---
        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        future_cpi = cpi_model.predict(future_years - 2010)

        # Center future CPI for benefits prediction
        future_cpi_centered = future_cpi - cpi_mean
        future_predictions_benefits = cpi_benefits_model.predict(future_cpi_centered.reshape(-1, 1))

        # --- Metrics for CPI vs Benefits regression ---
        mse_cpi = mean_squared_error(y_benefits, y_benefits_pred)
        r2_cpi = r2_score(y_benefits, y_benefits_pred)

        print("\n===========================================================================\n")
        print(f"[CPI vs Benefit Payments] Mean Squared Error: {mse_cpi}")
        print(f"[CPI vs Benefit Payments] R^2 Score: {r2_cpi}")
        print(f"[CPI vs Benefit Payments] Coefficient (slope): {cpi_benefits_model.coef_[0]:,.0f} $ / CPI Point")
        print(f"[CPI vs Benefit Payments] Intercept: ${cpi_benefits_model.intercept_:,.0f}\n")

        # Print forecast results (with commas in $ formatting)
        for year, pred in zip(future_years.ravel(), future_predictions_benefits):
            print(f"[CPI] Year: {year}, Predicted Benefit Payments: ${pred:,.0f}")

        # --- Plot: ONLY CPI vs Benefits relationship ---
        plt.figure(figsize=(10, 6))
        plt.scatter(X_cpi.ravel(), y_benefits/1000, label='Historical Benefit Payments')
        plt.plot(X_cpi.ravel(), y_benefits_pred/1000, linewidth=2, label='CPI Regression Line', color='green')
        plt.xlabel('Consumer Price Index')
        plt.ylabel('Benefit Payments in Thousand USD ($)')
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}".format(int(x))))
        plt.title('Benefit Payments vs. Consumer Price Index')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        # Store forecasts
        forecasts['cpi'] = future_predictions_benefits

        input("\nPress Enter to continue...")
        plt.close()


    if vars['income']:
        # --- Check column existence ---
        if 'Median Household Income' not in df.columns:
            raise ValueError("Column 'Median Household Income' is missing.")

        # --- Fit model for Income over Year ---
        X_income_year = df[['Year']].to_numpy()
        y_income = df['Median Household Income'].to_numpy()
        income_model = LinearRegression().fit(X_income_year - 2010, y_income)
        y_income_pred = income_model.predict(X_income_year - 2010)

        # --- Fit model for Benefit Payments vs Income ---
        X_income = df[['Median Household Income']].to_numpy()
        y_benefits = df['Benefit Payments'].to_numpy()

        # Optionally center/scaling to improve coefficients
        X_income_centered = X_income - X_income.mean()
        income_benefits_model = LinearRegression().fit(X_income_centered, y_benefits)
        y_benefits_pred = income_benefits_model.predict(X_income_centered)

        # --- Metrics ---
        mse_income = mean_squared_error(y_benefits, y_benefits_pred)
        r2_income = r2_score(y_benefits, y_benefits_pred)

        print("\n===========================================================================\n")
        print(f"[Income vs. Benefits Payments] Mean Squared Error: {mse_income}")
        print(f"[Income vs. Benefits Payments] R^2 Score: {r2_income}")
        print(f"[Income vs. Benefits Payments] Coefficients: {income_benefits_model.coef_[0]:,.0f} $ / $")
        print(f"[Income vs. Benefits Payments] Intercept: {income_benefits_model.intercept_:,.0f}\n")

        # --- Forecast future Income and Benefits ---
        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        future_income = income_model.predict(future_years - 2010)
        future_X_income_centered = future_income.reshape(-1, 1) - X_income.mean()
        future_benefits_pred = income_benefits_model.predict(future_X_income_centered)

        for year, pred in zip(future_years.ravel(), future_benefits_pred):
            print(f"[Income vs. Benefits Payments] Year: {year}, Predicted Benefit Payments: ${pred:,.0f}")

        # --- Plot: Benefit Payments vs Median Household Income (in thousands) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(X_income.ravel()/1000, y_benefits/1000, label='Actual Data')
        plt.plot(X_income.ravel()/1000, y_benefits_pred/1000, linewidth=2, label='Regression Line', color='green')
        plt.xlabel('Median Household Income (Thousands $)')
        plt.ylabel('Benefit Payments (Thousands $)')
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:,}".format(int(x))))

        plt.title('Benefit Payments vs Median Household Income')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # --- Store in forecasts ---
        forecasts['income'] = future_benefits_pred

        input("\nPress Enter to continue...")
        plt.close()



    if vars['snap']:
        # --- Check column existence ---
        if 'SNAP Recipients' not in df.columns:
            raise ValueError("Column 'SNAP Recipients' is missing.")

        print("Running SNAP Recipients...")

        # --- Use SNAP Recipients as predictor only for linear regression ---
        X_snap = df[['SNAP Recipients']].to_numpy()  # 1D predictor
        y_benefits = df['Benefit Payments'].to_numpy()

        snap_model = LinearRegression().fit(X_snap, y_benefits)
        y_benefits_pred = snap_model.predict(X_snap)

        # --- Metrics ---
        mse_snap = mean_squared_error(y_benefits, y_benefits_pred)
        r2_snap = r2_score(y_benefits, y_benefits_pred)
        print("\n===========================================================================\n")
        print(f"[SNAP vs. Benefit Payments] Mean Squared Error: {mse_snap:,.0f}")
        print(f"[SNAP vs. Benefit Payments] R^2 Score: {r2_snap:.4f}")
        print(f"[SNAP vs. Benefit Payments] Coefficient: SNAP: ${snap_model.coef_[0]:,.0f}")
        print(f"[SNAP vs. Benefit Payments] Intercept: ${snap_model.intercept_:,.0f}\n")

        # --- Forecast future Benefit Payments (keep using last SNAP value) ---
        future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
        last_snap = df['SNAP Recipients'].iloc[-1]
        future_X_snap = np.full(future_years.shape, last_snap)  # SNAP only
        future_predictions_snap = snap_model.predict(future_X_snap)

        for year, pred in zip(future_years.ravel(), future_predictions_snap):
            print(f"[SNAP vs. Benefit Payments] Year: {year}, Predicted Benefit Payments: ${pred:,.0f}")

        # --- Plot: Benefit Payments vs SNAP Recipients ---
        plt.figure(figsize=(10, 6))
        plt.scatter(df['SNAP Recipients'], y_benefits/1000, label='Actual Data')
        plt.plot(df['SNAP Recipients'], y_benefits_pred/1000, color='red', label='Linear Regression')
        plt.xlabel('SNAP Recipients')
        plt.ylabel('Benefit Payments in Thousand USD ($)')
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.title('Benefit Payments vs SNAP Recipients')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Store in forecasts ---
        forecasts['snap'] = future_predictions_snap

        input("\nPress Enter to continue...")
        plt.close()

    # --- Print combined future predictions ---
    print("\n===========================================================================\n")
    print("Combined Future Predictions for Benefit Payments:")
    print("\n===========================================================================\n")


    # --- Compute weighted combined forecast ---
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

    print("\n===========================================================================\n")
    
    for year, pred in zip(future_years.ravel(), combined_forecast):
        print(f"[Combined Model] Year: {year}, Predicted Benefit Payments: {pred:,.0f}")


    

    # --- Plot: only actual data + future predictions ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Year'], df['Benefit Payments'], label='Actual Data', color='blue')
    plt.scatter(future_years.ravel(), combined_forecast, marker='x', s=100, label='Weighted Future Predictions', color='purple')
    plt.xlabel('Year')
    plt.ylabel('Benefit Payments')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:}".format(int(x))))
    plt.title('Benefit Payments Forecasting (Weighted Future Predictions)')
    plt.legend()

    for x, y in zip(future_years.ravel(), combined_forecast):
        plt.text(x, y, f"{y:,.0f}", fontsize=9, ha='right', va='bottom', color='purple')

    plt.grid(True)
    plt.show()

    input("\nPress Enter to continue...")
    plt.close()

    # --- Save forecasted data ---
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

    print("Continue to Linear Regression Eligibles vs. Benefit Payments? (y/n): ", end="")
    cont = input().strip().lower()

    if cont != 'y' and cont != 'yes' and cont != 'Y' and cont != 'YES':
        print("Exiting program.")
        return

    clear_screen()

    eligibles_vs_payment(df)

    print("\n\nContinue to multiple regression? (y/n): ", end="")
    cont = input().strip().lower()

    if cont != 'y' and cont != 'yes' and cont != 'Y' and cont != 'YES':
        print("Exiting program.")
        return

    clear_screen()

    create_forecast_multiple(df)

    print("\nClosing...")

if __name__ == "__main__":
    main()