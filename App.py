import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Set the title of the web app
st.title("📊 Stock Market Analysis 📈 + Prediction using LSTM")

# Display an image for the online resource
st.image("https://t4.ftcdn.net/jpg/04/70/96/91/360_F_470969127_8pRb4Bl2Y7IxcA7cn7RHda7WhYIOb1Q6.jpg")

# Sidebar inputs for start date, end date, and stock ticker
st.sidebar.title('Stock Market Data')
min_date = datetime(2020, 1, 1)
max_date = datetime(2025, 12, 31)

start_date = st.sidebar.date_input(
    'Select a start date',
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    'Select an end date',
    value=max_date,
    min_value=start_date if start_date else min_date,
    max_value=max_date
)

# List of available stock tickers for selection
stock_LIST =[
    'ADANIENT.NS' , 'ADANIPORTS.NS' , 'APOLLOHOSP.NS' , 'ASIANPAINT.NS' , 'AXISBANK.NS' , 'BAJAJ-AUTO.NS' ,
    'BAJFINANCE.NS' , 'BAJAJFINSV.NS' , 'BPCL.NS' , 'BHARTIARTL.NS' , 'BRITANNIA.NS' , 'CIPLA.NS' , 'COALINDIA.NS' ,
    'DIVISLAB.NS' , 'DRREDDY.NS' , 'EICHERMOT.NS' , 'GRASIM.NS' , 'HCLTECH.NS' , 'HDFCBANK.NS' , 'HDFCLIFE.NS' ,
    'HEROMOTOCO.NS' , 'HINDALCO.NS' , 'HINDUNILVR.NS' , 'ICICIBANK.NS' , 'ITC.NS' , 'INDUSINDBK.NS' , 'INFY.NS' ,
    'JSWSTEEL.NS' , 'KOTAKBANK.NS' , 'LTIM.NS' , 'LT.NS' , 'M&M.NS' , 'MARUTI.NS' , 'NTPC.NS' , 'NESTLEIND.NS' ,
    'ONGC.NS' , 'POWERGRID.NS' , 'RELIANCE.NS' , 'SBILIFE.NS' , 'SHRIRAMFIN.NS' , 'SBIN.NS' , 'SUNPHARMA.NS' ,
    'TCS.NS' , 'TATACONSUM.NS' , 'TATAMOTORS.NS' , 'TATASTEEL.NS' , 'TECHM.NS' , 'TITAN.NS' , 'ULTRACEMCO.NS' ,
    'WIPRO.NS' ,
]
stock = st.sidebar.selectbox('Select stock ticker', stock_LIST)

# Function to fetch stock data from Yahoo Finance
@st.cache
def load_data(stock, start, end):
    try:
        data = yf.download(stock, start=start, end=end)
        return data
    except Exception as e:
        st.error(f'Error fetching data for {stock}: {e}')
        return None

# Calculate average daily return function
def calculate_average_daily_return(df):
    df['Daily Return'] = df['Adj Close'].pct_change()
    average_daily_return = df['Daily Return'].mean()
    return average_daily_return

# Display stock data if valid inputs are provided
if start_date and end_date and stock:
    st.write(f'Fetching data for {stock} from {start_date} to {end_date}...')
    df = load_data(stock, start_date, end_date)
    if df is not None and not df.empty:
        st.write(f'Data for {stock} from {start_date} to {end_date}')
        st.dataframe(df)

        # Calculate average daily return
        average_daily_return = calculate_average_daily_return(df)
        st.subheader(f'Average Daily Return of {stock}')
        st.write(f'{average_daily_return:.5f}')

        # Plotting historical closing prices and moving averages
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plotting Closing Prices and Moving Averages
        ax1.plot(df.index, df['Adj Close'], label='Adj Close', color='blue')
        ma_day = [10, 20, 50]
        for ma in ma_day:
            column_name = f"MA for {ma} days"
            df[column_name] = df['Adj Close'].rolling(ma).mean()
            ax1.plot(df.index, df[column_name], label=column_name)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'Historical Closing Prices and Moving Averages for {stock}')
        ax1.legend(loc='upper left')

        # Creating a secondary y-axis for Daily Return
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Daily Return'], label='Daily Return', color='red')
        ax2.set_ylabel('Daily Return', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Adding a horizontal line at y=0 for reference
        ax2.legend(loc='upper right')

        # Display average daily return on the graph
        ax2.annotate(f'Avg Daily Return: {average_daily_return:.5f}',
                     xy=(0.95, 0.95),
                     xycoords='axes fraction',
                     fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.5),
                     )

        st.pyplot(fig)
        
    elif df is not None and df.empty:
        st.write(f'No data available for {stock} from {start_date} to {end_date}')
else:
    st.write('Please enter all required fields.')

# Example of additional analysis (e.g., description and info for selected stock)
if stock:
    ticker = yf.Ticker(stock)
    description = ticker.info.get('longBusinessSummary', 'No description available')
    st.subheader(f"Description of {stock}")
    st.write(description)

    st.subheader(f"Info for {stock}")
    st.write(df.describe())

    # Plotting closing prices and moving averages
    st.subheader("Historical Closing Prices and Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df.index, df['Adj Close'], label='Adj Close')
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        ax.plot(df.index, df[column_name], label=column_name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Historical Closing Prices and Moving Averages for {stock}')
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Please enter a valid stock ticker and date range.")

# Risk vs. Expected Return Scatter Plot
st.title('Risk vs. Expected Return Scatter Plot')

# Sample Data for Demonstration (Replace this with your actual data loading)
# For example, you could load your data from a CSV
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

# For demonstration, creating a sample DataFrame
np.random.seed(0)
dates = pd.date_range('2021-01-01', periods=100)
sample_df = pd.DataFrame(np.random.randn(100, 4), index=dates, columns=['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D'])

# Calculate daily returns for the sample data
rets = sample_df.pct_change().dropna()

# Plotting
area = np.pi * 20  # Size of the points

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(rets.mean(), rets.std(), s=area)
ax.set_xlabel('Expected return')
ax.set_ylabel('Risk')
ax.set_title('Risk vs. Expected Return')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    ax.annotate(label,
                xy=(x, y),
                xytext=(50, 50),
                textcoords='offset points',
                ha='right',
                va='bottom',
                arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

ax.grid(True)
st.pyplot(fig)

# Analyzing correlation between stocks' closing prices
st.header('Correlation Analysis')
tech_list = ['AAPL', 'AMZN', 'GOOG', 'MSFT']  # Example stock tickers
start = '2022-01-01'
end = '2022-12-31'

closing_df = yf.download(tech_list, start=start, end=end)['Adj Close']
tech_rets = closing_df.pct_change().dropna()

st.subheader('Daily Returns Correlation')
st.dataframe(tech_rets.corr())

st.subheader('Pairplot of Daily Returns')
sns.pairplot(tech_rets, kind='reg')
st.pyplot(plt)

st.subheader('Heatmap of Daily Returns Correlation')
plt.figure(figsize=(12, 10))
sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock return')
st.pyplot(plt)

st.subheader('Heatmap of Closing Prices Correlation')
plt.figure(figsize=(12, 10))
sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock closing price')
st.pyplot(plt)

# Section for LSTM prediction
st.header('Apple Inc. Stock Price Prediction')

# Function to load the data for AAPL
@st.cache
def load_aapl_data():
    yf.pdr_override()
    df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
    return df

# Load the data
aapl_data = load_aapl_data()

# Plot the closing price history
st.subheader('Close Price History')
fig, ax = plt.subplots(figsize=(16,6))
ax.plot(aapl_data['Close'])
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(fig)

# Prepare the data for the LSTM model
st.subheader('Preparing Data for the Model')
df = aapl_data.filter(['Close'])
dataset = df.values
training_data_len = int(np.ceil(len(dataset) * .95))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plot the data
train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions

st.subheader('Model Predictions')
fig2, ax2 = plt.subplots(figsize=(16,6))
ax2.plot(train['Close'])
ax2.plot(valid[['Close', 'Predictions']])
ax2.set_xlabel('Date', fontsize=18)
ax2.set_ylabel('Close Price USD ($)', fontsize=18)
ax2.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig2)

# Show the valid and predicted prices
st.subheader('Valid and Predicted Prices')
st.write(valid)
