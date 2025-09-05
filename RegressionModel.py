import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clear_screen():
    # Check if the operating system is Windows ('nt')
    if os.name == 'nt':
        os.system('cls')  # Command for Windows
    else:
        os.system('clear') # Command for Linux/macOS


def toDF():

    # Load the Excel Spreadsheet, `madisonData.xlsx` into a DataFrame catch error if file not found
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
    # Display a summary of the DataFrame
    print("Original DataFrame Summary:")
    print(df.head())
    print()
    print(df.info())
    # print(df.describe(include='all'))


def create_forecast(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Check columns
    if 'Year' not in df.columns or 'Madison Eligibles' not in df.columns:
        raise ValueError("Required columns 'Year' and 'Madison Eligibles' are missing.")

    # --- Fit with NumPy arrays (no feature-name expectations) ---
    X = df[['Year']].to_numpy()                 # shape (n, 1)
    y = df['Madison Eligibles'].to_numpy()      # shape (n,)

    model = LinearRegression().fit(X, y)        # <-- only fit once
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Coefficients: {model.coef_}" + " (ie. Slope of line, impact of Year on Eligibles)")
    print(f"Intercept: {model.intercept_}\n")

    # --- Forecast future years as NumPy, keep flatten() working ---
    future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 6).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    print("Future Predictions for Number of Eligibles in Madison County:")
    for year, pred in zip(future_years.ravel(), future_predictions):
        print(f"Year: {year}, Predicted Eligibles: {pred:.2f}")

    

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X.ravel(), y, label='Actual Data')
    plt.plot(X.ravel(), y_pred, linewidth=2, label='Regression Line')
    plt.scatter(future_years.ravel(), future_predictions, marker='x', s=100, label='Future Predictions')
    plt.xlabel('Year')
    plt.ylabel('Madison Eligibles')
    plt.title('Madison Eligibles Forecasting')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save forecast to CSV, round to whole numbers
    forecast_df = pd.DataFrame({
        'Year': future_years.ravel(),
        'Predicted Madison Eligibles': np.round(future_predictions).astype(int)
    })

    

    forecast_df.to_csv('madison_forecast.csv', index=False)
    print("Forecasted data saved to 'madison_forecast.csv'.")


# Main function to run the script
def main():
    clear_screen()
    print("--- Madison County Eligibles Forecasting ---\n")
    df = toDF()
    # generate_summary(df)
    print()
    create_forecast(df)

if __name__ == "__main__":
    main()