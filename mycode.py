import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np # Ensure numpy is imported for scikit-learn compatibility

# --- Configuration & Caching ---

# Basic list of popular US stock symbols for the dropdown
# Note: You can expand this list or load it from an external source for production use
TICKER_OPTIONS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'NVDA', 'V', 'PYPL', 'NFLX', 'DIS', 'WMT']

# Set page configuration
st.set_page_config(
    page_title="Live Market & Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use Streamlit caching to speed up data fetching
# ttl=3600 caches the result for 1 hour, minimizing API calls
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period='1y'):
    """Fetches historical stock data from yfinance."""
    try:
        # Use a short period like '5d' or '1mo' for a faster initial load, 
        # but '1y' is fine for a broad chart.
        data = yf.download(ticker, period=period, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}. Please check the symbol and internet connection: {e}")
        return pd.DataFrame()

# --- Market Status Function ---

def check_market_status():
    """Checks if the US stock market (NYSE/NASDAQ) is currently open."""
    # Note: Using UTC for a universal check, assuming standard US trading hours (9:30 AM - 4:00 PM ET)
    now_utc = datetime.utcnow()
    
    # Check if it is a weekend (Saturday=5, Sunday=6)
    if now_utc.weekday() >= 5:
        return "**Market is CLOSED.** (Weekend)"
    
    # US trading hours: 9:30 AM ET (13:30 UTC) to 4:00 PM ET (20:00 UTC)
    
    current_hour_utc = now_utc.hour
    current_minute_utc = now_utc.minute

    market_open_utc = 13 + (30 / 60) # 13:30 UTC
    market_close_utc = 20 # 20:00 UTC

    current_time_in_hours = current_hour_utc + (current_minute_utc / 60)

    if market_open_utc <= current_time_in_hours < market_close_utc:
        return "**Market is LIVE and OPEN!** 🟢"
    else:
        return "**Market is CLOSED.** (After-Hours/Pre-Market)"

# --- Prediction Logic (Placeholder/Simple Example) ---

def simple_prediction_model(data, days_to_predict=5):
    """
    A simple Linear Regression model for prediction, serving as a placeholder.
    """
    if len(data) < 30:
        return None, "Not enough historical data (at least 30 points) for prediction."

    df = data[['Close']].copy()
    
    # The target is the price of the next day (shift(-1))
    df['Prediction_Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # Use the number of days passed as the primary feature (simple time trend)
    X = np.array(df.index.factorize()[0]).reshape(-1, 1) 
    y = df['Prediction_Target'].values

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next 'days_to_predict' values
    last_index = X[-1][0]
    future_indices = np.array(range(last_index + 1, last_index + 1 + days_to_predict)).reshape(-1, 1)
    
    future_predictions = model.predict(future_indices)
    
    # Create a DataFrame for the prediction
    last_date = df.index[-1].date()
    # Ensure prediction dates are successive business days (or just calendar days for simplicity here)
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    prediction_df = pd.DataFrame(
        future_predictions, 
        index=future_dates, 
        columns=['Predicted Close']
    )
    
    return prediction_df, None

# --- UI Layout ---

def main_app():
    """The main Streamlit application function."""
    st.title("📈 Live Market & Stock Price Predictor")
    
    # Display Market Status prominently
    status = check_market_status()
    st.markdown(f"### Market Status: {status}")
    st.markdown("---")

    # Sidebar for Stock Selection and Parameters (Feature 1: Dropdown)
    st.sidebar.header("Stock Selection & Settings")
    selected_ticker = st.sidebar.selectbox(
        'Select Stock Symbol',
        options=TICKER_OPTIONS,
        index=TICKER_OPTIONS.index('AAPL')
    )
    
    historical_period = st.sidebar.selectbox(
        'Historical Data Period',
        options=['1y', '3mo', '6mo', 'max', '5d'],
        index=0
    )

    days_to_predict = st.sidebar.slider(
        'Days to Predict (Simple Model)',
        min_value=1, max_value=30, value=7
    )

    # --- Fetch Data ---
    # The @st.cache_data decorator handles memoization here
    data = fetch_stock_data(selected_ticker, period=historical_period)

    if data.empty:
        st.error("Could not load stock data. Please check the symbol and try again.")
        return 

    # --- Real-time/Current Info Display ---
    st.header(f"Live Information for **{selected_ticker}**")
    
    # Get the latest data point (which is the last market day's closing data)
    latest_data = data.iloc[-1]
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest Close Price", f"${latest_data['Close']:.2f}", delta=f"{latest_data['Close'] - latest_data['Open']:.2f}")
    col2.metric("Latest Volume", f"{latest_data['Volume']:,}")
    col3.metric("Daily High", f"${latest_data['High']:.2f}")
    col4.metric("Daily Low", f"${latest_data['Low']:.2f}")
    
    st.markdown("---")

    # --- Charts & Share Info (Feature 2) ---
    
    # Historical Candlestick Chart
    st.subheader("Historical Candlestick Chart")
    
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])
    
    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        title=f"{selected_ticker} Price History ({historical_period})"
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Simple Prediction Section
    st.subheader(f"Price Prediction (Next {days_to_predict} Days)")
    prediction_df, error = simple_prediction_model(data, days_to_predict)

    if error:
        st.warning(f"Prediction Warning: {error}")
    elif prediction_df is not None:
        
        # Display the prediction as a table
        st.dataframe(prediction_df.style.format("${:.2f}"), use_container_width=True)

        # Plot the prediction
        fig_pred = go.Figure()

        # Plot historical close prices
        fig_pred.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'], 
            mode='lines', 
            name='Historical Close',
            line=dict(color='blue')
        ))
        
        # Plot predicted prices
        fig_pred.add_trace(go.Scatter(
            x=prediction_df.index, 
            y=prediction_df['Predicted Close'], 
            mode='lines+markers', 
            name='Predicted Close',
            line=dict(color='orange', dash='dash')
        ))

        fig_pred.update_layout(
            title=f'{selected_ticker} Simple Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)'
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.info("⚠️ **Prediction Note:** This forecast uses a basic Linear Regression model for illustration. **It is not reliable for real investment decisions.** A professional stock predictor requires an advanced, continuously trained Deep Learning model.")

    st.markdown("---")
    st.subheader("Raw Historical Data")
    # Display the last 10 rows
    st.dataframe(data.tail(10))


if __name__ == "__main__":
    main_app()
