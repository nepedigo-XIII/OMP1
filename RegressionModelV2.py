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

    print("\t      Part 2: Data Collection and Cleaning\n")

    print("==============================================================\n")

    print("\t> Madison County Eligibles <\t\n")
    
    print()

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

    print("==================================================================================\n")
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

    print("\n==================================================================================\n")
    print(f"Intercept: ${df.loc[df['Year'] == 2017, 'Benefit Payments'].iloc[0]:,.0f}")
    print(f"Slope (Coefficient): {slope:,.0f} $ / yr")
    print(f"Mean Squared Error: {mse:,.0f}")
    print(f"R^2 Score: {r2:.4f}")

    # Forecast 2023-2027
    future_years = pd.DataFrame({'Year': [2023, 2024, 2025, 2026, 2027]})
    future_predictions = model.predict(future_years)
    future_years['Predicted Benefit Payments'] = future_predictions

    # Print future predictions
    print("\n==================================================================================\n")
    print("\tFuture Year Predictions:\n")
    for _, row in future_years.iterrows():
        print(f"Year: {int(row['Year'])}\t|\t Predicted Benefit Cost: ${row['Predicted Benefit Payments']:,.2f}k")
    print("\n==================================================================================\n")

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
    


    

# def create_forecast_multiple(df):
#     from sklearn.linear_model import LinearRegression
#     from sklearn.metrics import mean_squared_error, r2_score
#     from sklearn.preprocessing import StandardScaler

#     vars = {
#             'year': True,
#             'unemployment': True,
#             'cpi': True,
#             'income': True,
#             'snap': True,  
#     }

#     weights = {
#         'year': 0.15,
#         'unemployment': 0.2,
#         'cpi': 0.2,
#         'income': 0.15,  
#         'snap': 0.3,    
#     }

#     forecasts = {}

#     if vars['year']:
        

#     if vars['unemployment']:
       
    
#     if vars['cpi']:
        

#     if vars['income']:
        

#     if vars['snap']:
        
        

#     # Final combination of forecasts
    

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