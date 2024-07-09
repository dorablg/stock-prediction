import os
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Determine the path to the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model
def load_stock_model():
    model_path = os.path.join(BASE_DIR, 'Stock Predictions Model.keras')
    model = load_model(model_path)
    return model

st.title('Stock Market Predictor')

st.sidebar.header('User Input')
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
start = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
end = st.sidebar.date_input('End Date', pd.to_datetime('2024-05-05'))

# Download stock data
data = yf.download(stock, start=start, end=end)

# Display stock data
st.subheader(f'Stock Data for {stock}')
st.write(data)

# Prepare data for prediction
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

# Scale data for prediction
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Visualizations
st.subheader('Price vs Moving Averages')

# Plot MA50 and MA100
fig1, ax1 = plt.subplots(figsize=(12, 8))
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ax1.plot(data.Close, label='Close Price', color='g')
ax1.plot(ma_50_days, label='50-Day MA', color='r')
ax1.plot(ma_100_days, label='100-Day MA', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

# Plot MA200
fig2, ax2 = plt.subplots(figsize=(12, 8))
ma_200_days = data.Close.rolling(200).mean()
ax2.plot(data.Close, label='Close Price', color='g')
ax2.plot(ma_200_days, label='200-Day MA', color='y')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)

# Prepare data for model prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)
model = load_stock_model()

# Perform predictions
predictions = model.predict(x)

# Inverse scale predictions and actual prices
scale = 1 / scaler.scale_[0]
predictions = predictions * scale
y = y * scale

# Plot original vs predicted prices
st.subheader('Original Price vs Predicted Price')
fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(y, label='Original Price', color='g')
ax3.plot(predictions, label='Predicted Price', color='r')
ax3.set_xlabel('Time')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)

# Automatic interpretation of stock predictions
st.subheader('Stock Value Prediction Interpretation')
future_days = st.sidebar.slider('Days of Future Prediction', 1, 60, 30)
future_predictions = predictions[-future_days:]

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(range(1, future_days + 1), future_predictions, label='Future Predictions', color='r')
ax4.set_xlabel('Days')
ax4.set_ylabel('Price')
ax4.set_title(f'Predicted Prices for Next {future_days} Days')
ax4.legend()
st.pyplot(fig4)

interpretation = (
    f"The model predicts an {'upward' if future_predictions[-1] > future_predictions[0] else 'downward'} trend "
    f"in the stock prices over the next {future_days} days."
)
st.write(interpretation)

# Key Prediction Points
st.subheader('Key Prediction Points')
key_points = {
    'Highest Predicted Price': np.max(future_predictions),
    'Lowest Predicted Price': np.min(future_predictions),
    'Predicted Price at the End': future_predictions[-1]
}

for key, value in key_points.items():
    if isinstance(value, np.ndarray):
        value = value.item()  # Convert ndarray to scalar if possible
    st.write(f"{key}: {value:.2f}")
