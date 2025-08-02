import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Hiding warnings to keep our output clean and focused.
warnings.filterwarnings("ignore")

# --- Project Configuration: Setting Up Our Analysis! ---
# This is where your stock data is located. Please ensure this path is correct!
FILE_PATH = r"C:\Users\jatin\OneDrive\Desktop\Zidio Internship\Data_analytics_Stock_market.xlsx"

# We'll predict 1 day into the future (tomorrow's closing price).
PREDICTION_DAYS = 1

# We'll use 80% of our data to train the model and 20% to test it.
TRAIN_SPLIT_RATIO = 0.8

# Setting a nice visual style for all our charts.
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# IMPORTANT: For a quick demo, we'll process only a few stocks.
# Set this to a small number (e.g., 3, 5) to try it out.
# If you want to analyze ALL stocks (be prepared, it takes time and opens many windows!),
# simply set MAX_STOCKS_TO_PROCESS = None.
MAX_STOCKS_TO_PROCESS = 5 # Analyzing 5 stocks for this presentation.

# --- 1. Data Loading and Preprocessing: Getting Our Data Ready! ---
def load_and_preprocess_data(file_path):
    """
    My first step was to load the stock data, clean it, and prepare it for analysis.
    """
    print(f"First, let's load the data from: {file_path}")
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded! Original size: {df.shape}")

        # Removing any rows with missing values to ensure clean data.
        df_cleaned = df.dropna().copy()
        print(f"Data cleaned! New size: {df_cleaned.shape}")

        # Converting the 'Date' column to a proper date format.
        df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
        # Setting 'Date' as the index for easier time-based analysis.
        df_cleaned = df_cleaned.set_index('Date')
        # Sorting data by date, which is crucial for time series.
        df_cleaned = df_cleaned.sort_index()

        print("Data is now clean and ready!")
        print("Here's a look at the first few rows:")
        print(df_cleaned.head())
        print("\nAnd a check on the data types:")
        df_cleaned.info()
        return df_cleaned

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- 2. Preparing Data for Prediction: Setting Up Inputs and Outputs! ---
def prepare_data_for_prediction(stock_df, target_column='Close', prediction_days=1):
    """
    This function prepares our data by creating a 'target' (what we predict)
    and 'features' (what we use to predict).
    """
    df_pred = stock_df.copy()

    # Our target: The closing price 'PREDICTION_DAYS' into the future.
    df_pred['Target_Close'] = df_pred[target_column].shift(-prediction_days)

    # Our main feature: The current day's closing price.
    df_pred['Feature_Current_Close'] = df_pred[target_column]

    # Removing rows where the target (future price) is missing.
    df_pred.dropna(inplace=True)

    # Defining our input (X) and output (y) for the model.
    X_features = df_pred[['Feature_Current_Close']]
    y_target = df_pred['Target_Close']

    # Splitting data chronologically: 80% for training, 20% for testing.
    train_data_size = int(len(df_pred) * TRAIN_SPLIT_RATIO)
    X_train_set, X_test_set = X_features[0:train_data_size], X_features[train_data_size:len(df_pred)]
    y_train_set, y_test_set = y_target[0:train_data_size], y_target[train_data_size:len(df_pred)]

    print(f"\nData is ready for prediction! Total samples: {len(df_pred)}")
    print(f"Training samples: {len(X_train_set)} | Testing samples: {len(X_test_set)}")

    return X_train_set, X_test_set, y_train_set, y_test_set, df_pred

# --- 3. Model Evaluation: How Well Did Our Model Perform? ---
def evaluate_model(model_name, y_true, y_pred):
    """
    This function calculates key metrics to assess our model's accuracy.
    """
    if len(y_true) == 0:
        print(f"  No data to evaluate for {model_name}.")
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    # Root Mean Squared Error (RMSE): Average prediction error in INR.
    rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
    # Mean Absolute Error (MAE): Average absolute difference between actual and predicted.
    mae_score = mean_absolute_error(y_true, y_pred)
    # R-squared (R2): How much variance in actual prices the model explains (closer to 1 is better).
    r2_score_val = r2_score(y_true, y_pred)

    print(f"\n--- Performance for {model_name} ---")
    print(f"  RMSE: {rmse_score:.4f} INR")
    print(f"  MAE: {mae_score:.4f} INR")
    print(f"  R-squared (R2): {r2_score_val:.4f}")
    return {'RMSE': rmse_score, 'MAE': mae_score, 'R2': r2_score_val}

# --- 4. Visualizations: Bringing Our Data and Predictions to Life! ---
def plot_all_summaries(stock_name, stock_df, y_test_lr, y_pred_lr, lr_future_pred):
    """
    This function creates ONE comprehensive plot window for all our insights
    for the selected stock.
    """
    print(f"Generating all visualizations for {stock_name} in one window...")
    fig, axes = plt.subplots(3, 1, figsize=(18, 20)) # 3 rows, 1 column for our plots
    fig.suptitle(f'Stock Analysis for {stock_name}', fontsize=28, y=0.98)
    plt.subplots_adjust(hspace=0.4) # Adding space between plots

    # Plot 1: Historical Close Price
    axes[0].plot(stock_df.index, stock_df['Close'], label='Actual Close Price', color='blue', linewidth=1.5)
    axes[0].set_title('1. Historical Close Price Over Time', fontsize=18)
    axes[0].set_ylabel('Price (INR)', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='x', rotation=30, labelsize=12)

    # Plot 2: Historical Trading Volume
    axes[1].plot(stock_df.index, stock_df['Volume'], label='Trading Volume', color='green', alpha=0.8, linewidth=1.5)
    axes[1].set_title('2. Historical Trading Volume Over Time', fontsize=18)
    axes[1].set_xlabel('Date', fontsize=14)
    axes[1].set_ylabel('Volume', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='x', rotation=30, labelsize=12)

    # Plot 3: Linear Regression - Actual vs. Predicted with Future Forecast
    axes[2].plot(y_test_lr.index, y_test_lr, label='Actual Price', color='blue', alpha=0.7, linewidth=1.5)
    axes[2].plot(y_test_lr.index, y_pred_lr, label='Predicted Price', color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    if lr_future_pred is not None and not y_test_lr.empty:
        last_actual_date = y_test_lr.index[-1]
        last_actual_price = y_test_lr.iloc[-1]
        next_day_date = last_actual_date + pd.Timedelta(days=PREDICTION_DAYS)
        axes[2].plot([last_actual_date, next_day_date], [last_actual_price, lr_future_pred],
                     marker='o', markersize=8, color='purple', linestyle=':', label='Next Day Forecast')
        axes[2].text(next_day_date, lr_future_pred, f' {lr_future_pred:.2f}',
                     color='purple', va='bottom', ha='left', fontsize=9, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=0.5, alpha=0.6))
    axes[2].set_title('3. Linear Regression: Actual vs. Predicted & Future Forecast', fontsize=18)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Price (INR)', fontsize=12)
    axes[2].legend(fontsize=10, loc='upper left')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].tick_params(axis='x', rotation=30, labelsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show() # Display this single, comprehensive plot window!

# --- Main Execution Block: Let's Run the Show! ---
if __name__ == "__main__":
    print("Hello everyone, and welcome to my Data Analytics Internship Project!")
    print("Today, I'll be presenting a simplified analysis and forecasting of stock market trends.")

    # Step 1: Loading and preparing our data.
    main_cleaned_df = load_and_preprocess_data(FILE_PATH)
    if main_cleaned_df is None:
        print("Project cannot proceed without data. Exiting.")
        exit()

    # Get all unique stock names from the dataset.
    all_unique_stock_names = main_cleaned_df['Stock Name'].unique()
    print(f"\nWe've found {len(all_unique_stock_names)} unique stocks in this dataset!")

    # Deciding which stocks to analyze for this presentation.
    if MAX_STOCKS_TO_PROCESS is not None and MAX_STOCKS_TO_PROCESS > 0:
        stocks_to_process = all_unique_stock_names[:MAX_STOCKS_TO_PROCESS]
        print(f"For this demo, we'll focus on the first {len(stocks_to_process)} stocks.")
    else:
        stocks_to_process = all_unique_stock_names
        print("Alright, processing ALL stocks. This will take a while!")

    # Now, let's analyze each selected stock!
    for i, stock_name in enumerate(stocks_to_process):
        print(f"\n\n{'='*60}")
        print(f"Analyzing Stock: {stock_name} ({i+1} of {len(stocks_to_process)})")
        print(f"{'='*60}")

        # Filtering data for the current stock.
        stock_df = main_cleaned_df[main_cleaned_df['Stock Name'] == stock_name].copy()
        if stock_df.empty:
            print(f"No data found for {stock_name}. Skipping.")
            continue
        if len(stock_df) < 50: # We need enough data points for meaningful analysis.
            print(f"Not enough data points ({len(stock_df)}) for {stock_name}. Skipping analysis.")
            continue

        stock_df = stock_df.sort_index()

        # Step 2: Preparing data for our prediction model.
        X_train_lr, X_test_lr, y_train_lr, y_test_lr, df_for_future_pred = \
            prepare_data_for_prediction(stock_df, target_column='Close', prediction_days=PREDICTION_DAYS)

        if X_train_lr.empty or X_test_lr.empty:
            print(f"Not enough data for {stock_name} to train the model. Skipping prediction.")
            continue

        # Step 3: Training and Evaluating our Linear Regression Model.
        print(f"\n--- Training Linear Regression for {stock_name} ---")
        lr_model = LinearRegression()
        lr_model.fit(X_train_lr, y_train_lr)
        y_pred_lr = lr_model.predict(X_test_lr)
        lr_metrics = evaluate_model("Linear Regression", y_test_lr, y_pred_lr)

        # Making a future prediction with Linear Regression.
        lr_future_pred = None
        if not y_test_lr.empty:
            last_known_close = df_for_future_pred['Feature_Current_Close'].iloc[-1]
            future_prediction_input = pd.DataFrame([[last_known_close]], columns=['Feature_Current_Close'])
            lr_future_pred = lr_model.predict(future_prediction_input)[0]
            print(f"Linear Regression predicts next day close: {lr_future_pred:.2f} INR")
        else:
            print("Test set too small for future prediction.")

        # Step 4: Visualizing all our findings in one window!
        plot_all_summaries(stock_name, stock_df, y_test_lr, y_pred_lr, lr_future_pred)

    print("\n--- Project Analysis Complete! ---")
    print("Thank you for your attention. This project demonstrates foundational skills in data analysis and time series forecasting.")
    print("I'm happy to answer any questions you may have!")