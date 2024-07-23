import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Function to preprocess data
def preprocess_data(df):
    df = df.dropna()
    
    # Feature scaling
    scaler = StandardScaler()
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    
    X_scaled = scaler.fit_transform(X)
    
    # Use time-based split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test, scaler

# Function to train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

# Function to plot raw data
def plot_raw_data(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title('Stock Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot predictions
def plot_predictions(y_test, y_pred, y_test_dates):
    y_test_series = pd.Series(y_test.values, index=y_test_dates)
    y_pred_series = pd.Series(y_pred, index=y_test_dates)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(y_test_series.index, y_test_series.values, label='Actual', marker='o', color='blue')
    plt.plot(y_pred_series.index, y_pred_series.values, label='Predicted', marker='x', color='red')
    
    plt.title('Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit app
st.title('Stock Price Prediction')

# Load the dataset
df = pd.read_csv('HDFC.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sidebar for date range selection
st.sidebar.header('Select Date Range')
start_date = st.sidebar.date_input('Start Date', df.index.min())
end_date = st.sidebar.date_input('End Date', df.index.max())

# Filter data based on selected date range
filtered_df = df.loc[start_date:end_date]

# Ensure there's data to plot
if not filtered_df.empty:
    st.write("Data Preview:")
    st.write(filtered_df.head())

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(filtered_df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)

    # Save model
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Extract dates for plotting
    y_test_dates = filtered_df.index[-len(y_test):]

    # Show raw data plot
    st.subheader('Raw Data')
    plot_raw_data(filtered_df)

    # Show predictions plot
    st.subheader('Predictions')
    plot_predictions(y_test, y_pred, y_test_dates)

    # Show performance metrics
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R^2 Score: {r2}')

    # Provide download link for the trained model
    st.download_button(
        label="Download Trained Model",
        data=pickle.dumps(model),
        file_name="linear_regression_model.pkl"
    )
else:
    st.write("No data available for the selected date range.")
