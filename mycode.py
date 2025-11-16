import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import pytz
import locale

# Set locale for INR formatting (optional, but helps with currency display)
locale.setlocale(locale.LC_ALL, 'en_IN.utf8')

# --- Configuration & Caching ---

# Set page configuration
st.set_page_config(
    page_title="Indian Market Trend Dashboard (NSE/BSE)",
    layout="wide", # Use wide layout for dashboard look
    initial_sidebar_state="expanded"
)

# Utility to clean ticker for external URL use
def clean_ticker_for_url(ticker):
    """Cleans yfinance ticker (e.g., RELIANCE.NS) to base symbol (RELIANCE) for URL embedding."""
    return ticker.split('.')[0]

# Function to format large numbers (Mkt Cap, Volume)
def format_large_number(num):
    """Formats a large number into L, Cr, Tn units."""
    if num >= 1e12:
        return f"{num/1e12:.2f}Tn"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B" # Billions
    elif num >= 1e7:
        return f"{num/1e7:.2f}Cr" # Crores
    elif num >= 1e5:
        return f"{num/1e5:.2f}L" # Lakhs
    else:
        return f"{num:,.2f}"

# Use Streamlit caching to speed up data fetching
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stock_data(ticker, period='1y', interval='1d'):
    """Fetches historical stock data from yfinance."""
    if not ticker or ticker.strip() == "":
        return pd.DataFrame()
        
    try:
        # Check if the ticker is numeric (BSE code) and needs .BO suffix
        if ticker.isdigit() and not ticker.endswith('.BO'):
            full_ticker = ticker + '.BO'
        else:
            full_ticker = ticker
            
        data = yf.download(full_ticker, period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}. Please ensure the ticker is correct. Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_ticker_info(ticker):
    """Fetches fundamental data for the ticker."""
    try:
        # Check if the ticker is numeric (BSE code) and needs .BO suffix
        if ticker.isdigit() and not ticker.endswith('.BO'):
            full_ticker = ticker + '.BO'
        else:
            full_ticker = ticker
            
        info = yf.Ticker(full_ticker).info
        return info
    except Exception as e:
        # Lowered to a debug message so the app doesn't break if info fails
        print(f"Warning: Could not fetch fundamental data for {ticker}. Error: {e}") 
        return {}

@st.cache_data(ttl=600)
def fetch_stock_news(ticker):
    """Fetches real-time news for the ticker from yfinance."""
    if not ticker:
        return []
    try:
        # Handle BSE vs NSE format
        if ticker.isdigit() and not ticker.endswith('.BO'):
            full_ticker = ticker + '.BO'
        else:
            full_ticker = ticker
            
        t = yf.Ticker(full_ticker)
        news_data = t.news
        return news_data
    except Exception as e:
        print(f"Warning: Could not fetch news for {ticker}. Error: {e}")
        return []


# --- Technical Analysis Functions (Existing - Kept for TA Tab) ---

def calculate_indicators(data):
    """Calculates EMA, RSI, and MACD."""
    df = data.copy()
    
    # EMAs
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
    df['EMA_15_Offset_5'] = df['EMA_15'].shift(-5)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    df = df.drop(columns=['EMA_12', 'EMA_26'], errors='ignore')
    
    return df

def generate_decision(data_row):
    """Generates a Buy, Hold, or Sell decision (using scalar values)."""
    if data_row.empty:
        return "HOLD", "orange", "Insufficient data for a technical decision.", ["Data not loaded."]

    clean_data = data_row.dropna()
    
    if clean_data.empty:
        return "HOLD", "orange", "Indicators require more historical data (min 26 days).", ["Data Incomplete"]

    latest = clean_data.iloc[-1]
    
    try:
        # CRITICAL FIX: Explicitly extract the scalar values using .item()
        close = latest['Close'].item()
        ema_5 = latest['EMA_5'].item()
        ema_15 = latest['EMA_15'].item()
        macd = latest['MACD'].item()
        signal_line = latest['Signal_Line'].item()
        rsi = latest['RSI'].item()
    except Exception:
        return "HOLD", "red", "Internal data extraction error.", ["Error extracting latest data."]
    
    score = 0
    rationale = []

    # 1. Price Action
    if close > ema_5 and ema_5 > ema_15:
        score += 2
        rationale.append(f"Uptrend: Close ({close:.2f}) > 5 EMA ({ema_5:.2f}) > 15 EMA ({ema_15:.2f}).")
    elif close < ema_15:
        score -= 2
        rationale.append(f"Downtrend: Close ({close:.2f}) is below 15 EMA ({ema_15:.2f}).")
    else:
        rationale.append("Mixed EMA signals.")

    # 2. Momentum (MACD)
    if macd > signal_line:
        score += 1
        rationale.append("MACD is bullish (MACD Line > Signal Line).")
    elif macd < signal_line:
        score -= 1
        rationale.append("MACD is bearish (MACD Line < Signal Line).")
    
    # 3. Strength (RSI)
    if rsi < 30:
        score += 1
        rationale.append(f"RSI ({rsi:.2f}) is oversold (< 30).")
    elif rsi > 70:
        score -= 1
        rationale.append(f"RSI ({rsi:.2f}) is overbought (> 70).")
    
    
    if score >= 2:
        decision, color, summary = "BUY", "green", "Strong upward pressure."
    elif score <= -2:
        decision, color, summary = "SELL", "red", "Immediate weakness and downtrend are visible."
    else:
        decision, color, summary = "HOLD", "orange", "Indicators are mixed or neutral (consolidation)."

    return decision, color, summary, rationale

# --- Mock Market Data Function (Existing) ---

@st.cache_data(ttl=600)
def mock_fetch_gainers_losers(n=15):
    """Mocks the fetching of Top N Gainers and Losers."""
    
    all_tickers = [
        'HINDCOPPER.NS', 'ZOMATO.NS', 'TATASTEEL.NS', 'SAIL.NS', 'JSWSTEEL.NS', 
        'INFY.NS', 'TCS.NS', 'SBIN.NS', 'RELIANCE.NS', 'WIPRO.NS', 
        'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'VEDL.NS', 'NTPC.NS',
        'BEL.NS', 'HDFCLIFE.NS', 'MINDTREE.NS', 'LT.NS', 'BAJFINANCE.NS', 
        'TITAN.NS', 'HEROMOTOCO.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'DRREDDY.NS',
        'ADANIPORTS.NS', 'NESTLEIND.NS', 'TATACONSUM.NS', 'SUNPHARMA.NS', 'HCLTECH.NS', 
        'INDUSINDBK.NS', 'KOTAKBANK.NS', 'POWERGRID.NS', 'M&M.NS', 'TATACOMM.NS'
    ]
    
    try:
        mock_stocks = np.random.choice(all_tickers, size=n * 2, replace=False)
    except ValueError:
        return pd.DataFrame(), pd.DataFrame()
    
    gainers_data = []
    losers_data = []

    for i in range(n):
        gain = np.random.uniform(2.5, 10.0)
        price_g = np.random.uniform(500, 3500)
        gainers_data.append({
            'Symbol': mock_stocks[i].replace(".NS", ""),
            'LTP (₹)': price_g,
            'Chg (₹)': price_g * (gain / 100),
            '% Chg': gain
        })

        loss = np.random.uniform(-10.0, -2.5)
        price_l = np.random.uniform(500, 3500)
        losers_data.append({
            'Symbol': mock_stocks[i + n].replace(".NS", ""),
            'LTP (₹)': price_l,
            'Chg (₹)': price_l * (loss / 100),
            '% Chg': loss
        })
    
    gainers_df = pd.DataFrame(gainers_data)
    losers_df = pd.DataFrame(losers_data)
    
    gainers_df = gainers_df.sort_values(by='% Chg', ascending=False)
    losers_df = losers_df.sort_values(by='% Chg', ascending=True)
    
    return gainers_df.head(n), losers_df.head(n)

# --- Market Status Function (IST/NSE) ---

def check_market_status():
    """Checks if the Indian stock market (NSE) is currently open."""
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    if now_ist.weekday() >= 5:
        return "Weekend"
    market_open_ist = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_ist = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    if market_open_ist <= now_ist < market_close_ist:
        return "LIVE and OPEN"
    else:
        return "Closed"

# --- UI Layout ---

def main_app():
    """The main Streamlit application function."""
    
    # --- Sidebar for Stock Selection and Parameters ---
    st.sidebar.header("Stock Selection & Settings")
    
    selected_ticker_input = st.sidebar.text_input(
        'Enter Stock Symbol (NSE/BSE)',
        value='RELIANCE.NS', # Changed default to a highly reliable ticker
        placeholder='e.g., RELIANCE.NS or 500325'
    ).strip().upper() 
    
    # Auto-adjust ticker for fetching if it looks like a BSE code
    if selected_ticker_input.isdigit() and len(selected_ticker_input) == 6:
        selected_ticker = selected_ticker_input + '.BO'
    else:
        selected_ticker = selected_ticker_input

    st.sidebar.markdown("""
    **Format Guidance:**
    * **NSE:** Ticker followed by `.NS` (e.g., `TCS.NS`)
    * **BSE:** The 6-digit Security Code (e.g., `500325`)
    """)
    
    # Data Period selection - controls the scope of data fetched
    historical_period = st.sidebar.selectbox(
        'Historical Data Period (Scope)',
        options=['1D', '5D', '1M', '6M', 'YTD', '1Y', '5Y', 'Max'],
        index=0,
        format_func=lambda x: x # Use the selected value as is
    ).lower() # yfinance expects lowercase period strings

    chart_interval = st.sidebar.selectbox(
        'Chart Timeframe (Candle Size)',
        options=['1d', '1wk', '1mo', '1h', '30m', '15m', '5m', '1m'], 
        index=0
    )

    if not selected_ticker:
        st.info("Please enter a stock symbol in the sidebar to load the data.")
        return

    # --- Fetch Data & Indicators ---
    # Convert 'ytd' to '1y' or handle it, as yfinance period is simple. We'll use 1Y for YTD for now.
    period_yf = historical_period if historical_period != 'ytd' else '1y' 
    data = fetch_stock_data(selected_ticker, period=period_yf, interval=chart_interval)
    ticker_info = fetch_ticker_info(selected_ticker)

    if data.empty:
        st.error(f"Could not load data for {selected_ticker}. Please check the symbol and try again.")
        return 
    
    data_with_indicators = calculate_indicators(data)
    display_name = ticker_info.get('longName', selected_ticker.split('.')[0]) # Use long name if available

    # --- Dashboard Tabs ---
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Overview", 
        "📊 Technical Analysis",
        "📈 Market Movers",
        "📰 Real-Time News", # New News Tab
        "🔮 Price Prediction" # Prediction is now the 5th tab
    ])
    
    # Get latest/previous close data
    latest_close = data['Close'].iloc[-1].item()
    
    if len(data) > 1:
        previous_close = data['Close'].iloc[-2].item()
    else:
        previous_close = ticker_info.get('regularMarketPreviousClose', latest_close)

    price_change = latest_close - previous_close
    percent_change = (price_change / previous_close) * 100 if previous_close else 0
    
    change_color = "green" if price_change >= 0 else "red"
    change_symbol = "▲" if price_change >= 0 else "▼"
    market_status = check_market_status()
    
    # --- TAB 1: Overview ---
    with tab1:
        
        # 1. HEADER
        st.subheader(f"Market Summary > {display_name}")
        
        # 2. CURRENT PRICE & FOLLOW BUTTON
        col_price, col_follow = st.columns([1, 4])
        
        with col_price:
            # Current Price
            st.markdown(f"""
            <div style='line-height: 0.9; margin-bottom: 5px;'>
                <span style='font-size: 3.5em; font-weight: 600;'>{latest_close:.2f}</span>
                <span style='font-size: 1.5em; color: #555;'>INR</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Price Change
            st.markdown(f"""
            <span style='font-size: 1.2em; color: {change_color}; font-weight: 500;'>
                {change_symbol}{abs(price_change):.2f} ({abs(percent_change):.2f}%)
            </span>
            <span style='font-size: 0.9em; color: #555;'>
                today
            </span>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<p style='font-size: 0.8em; color: #777; margin-top: 5px;'>{datetime.now().strftime('%d %b, %I:%M %p IST')} • {market_status}</p>", unsafe_allow_html=True)


        with col_follow:
            # Dummy Follow Button styling
            st.markdown(
                f"""
                <style>
                    .stButton>button {{
                        background-color: transparent;
                        color: #1a73e8;
                        border: 1px solid #1a73e8;
                        border-radius: 8px;
                        padding: 8px 16px;
                        font-weight: 500;
                        margin-top: 25px;
                    }}
                </style>
                """, unsafe_allow_html=True
            )
            st.button("➕ Follow", key="follow_button")
        
        # 3. CHART AREA
        
        # Simple Line Chart for Overview Tab
        fig_overview = go.Figure()

        # Add Price Line (Close)
        fig_overview.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'], 
            mode='lines', 
            line=dict(color=change_color, width=2), 
            name='Close Price'
        ))
        
        # Add Previous Close marker/line (Horizontal Line)
        fig_overview.add_hline(
            y=previous_close, 
            line_dash="dot", 
            line_color="#777", 
            annotation_text=f"Previous close {previous_close:.2f}",
            annotation_position="right",
            annotation_font_color="#777",
            opacity=0.5
        )

        fig_overview.update_layout(
            height=400,
            showlegend=False,
            xaxis_title=None,
            yaxis_title=None,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=20)
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)


        # 4. KEY METRICS GRID (Mimicking the bottom section)
        st.markdown("<hr style='border: 1px solid #f0f0f0;'>", unsafe_allow_html=True)

        col_left, col_middle, col_right = st.columns(3)

        # Helper function to display metric block
        def display_metric(col, label, value):
            col.markdown(f"<p style='font-size: 0.8em; color: #777; margin-bottom: 2px;'>{label}</p><p style='font-size: 1.1em; font-weight: 500;'>{value}</p>", unsafe_allow_html=True)

        # LEFT COLUMN (Price Action)
        display_metric(col_left, "Open", f"₹{data['Open'].iloc[-1].item():.2f}")
        display_metric(col_left, "High", f"₹{data['High'].iloc[-1].item():.2f}")
        display_metric(col_left, "Low", f"₹{data['Low'].iloc[-1].item():.2f}")

        # MIDDLE COLUMN (Fundamental Ratios)
        # Fallback values if YF info failed
        mkt_cap = format_large_number(ticker_info.get('marketCap', 0) if ticker_info.get('marketCap') else 1.2e13) 
        pe_ratio = ticker_info.get('trailingPE', 28.5)
        div_yield = ticker_info.get('dividendYield', 0.012) * 100 if ticker_info.get('dividendYield') else 0.00
        
        display_metric(col_middle, "Mkt cap", mkt_cap)
        display_metric(col_middle, "P/E ratio", f"{pe_ratio:.2f}")
        display_metric(col_middle, "Div yield", f"{div_yield:.2f}%")
        
        # RIGHT COLUMN (Yearly/Other Metrics)
        wk52_high = ticker_info.get('fiftyTwoWeekHigh', data['High'].max())
        wk52_low = ticker_info.get('fiftyTwoWeekLow', data['Low'].min())
        qtrly_div = ticker_info.get('lastDividendValue', 0.0)

        display_metric(col_right, "52-wk high", f"₹{wk52_high:.2f}")
        display_metric(col_right, "52-wk low", f"₹{wk52_low:.2f}")
        display_metric(col_right, "Qtrly div amt", f"₹{qtrly_div:.2f}")
        
    # --- TAB 2: Advanced Technical Chart ---
    with tab2:
        st.header(f"{display_name} Advanced Technical Analysis")
        
        # Display Decision
        decision, color, summary, rationale_list = generate_decision(data_with_indicators)
        
        if color == "green":
            st.success(f"**Decision: {decision}** - {summary}")
        elif color == "red":
            st.error(f"**Decision: {decision}** - {summary}")
        else:
            st.warning(f"**Decision: {decision}** - {summary}")
            
        with st.expander("Show Detailed Rationale"):
            for point in rationale_list:
                st.markdown(f"- {point}")
        
        st.markdown("---")
        
        # Create the advanced multi-chart figure
        fig_ta = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.6, 0.13, 0.13, 0.14],
            subplot_titles=('Price & EMAs', 'Volume', 'RSI', 'MACD')
        )
        
        # Row 1: Candlestick Chart (Main) with EMAs
        fig_ta.add_trace(go.Candlestick(
            x=data_with_indicators.index, open=data_with_indicators['Open'],
            high=data_with_indicators['High'], low=data_with_indicators['Low'],
            close=data_with_indicators['Close'], name='Price'
        ), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['EMA_5'], 
                                 line=dict(color='blue', width=1), name='5 EMA'), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['EMA_15'], 
                                 line=dict(color='red', width=1), name='15 EMA'), row=1, col=1)
        
        # Row 2: Volume
        volume_color = np.where(data_with_indicators['Close'] > data_with_indicators['Open'], 'rgba(0,128,0,0.5)', 'rgba(255,0,0,0.5)')
        fig_ta.add_trace(go.Bar(x=data_with_indicators.index, y=data_with_indicators['Volume'], 
                             name='Volume', marker_color=volume_color), row=2, col=1)
        
        # Row 3: RSI
        fig_ta.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['RSI'], 
                                 line=dict(color='purple', width=1.5), name='RSI (14)'), row=3, col=1)
        fig_ta.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1, opacity=0.8)
        fig_ta.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1, opacity=0.8)
        
        # Row 4: MACD
        hist_color = np.where(data_with_indicators['MACD_Histogram'] > 0, 'green', 'red')
        fig_ta.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['MACD'], 
                                 line=dict(color='blue', width=1.5), name='MACD Line'), row=4, col=1)
        fig_ta.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Signal_Line'], 
                                 line=dict(color='orange', width=1.5), name='Signal Line'), row=4, col=1)
        fig_ta.add_trace(go.Bar(x=data_with_indicators.index, y=data_with_indicators['MACD_Histogram'], 
                             name='Histogram', marker_color=hist_color, opacity=0.6), row=4, col=1)

        fig_ta.update_layout(height=900, xaxis_rangeslider_visible=False)
        fig_ta.update_yaxes(title_text="Price", row=1, col=1)
        fig_ta.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig_ta.update_yaxes(title_text="MACD", row=4, col=1)
        
        st.plotly_chart(fig_ta, use_container_width=True)

    # --- TAB 3: Market Movers (Unchanged) ---
    with tab3:
        st.header("Top 15 Gainers and Losers (NSE/BSE Mock Data)")
        st.warning("Note: This data is generated using a **mock function**.")
        
        gainers_df_raw, losers_df_raw = mock_fetch_gainers_losers(n=15)
        
        col_gainer, col_loser = st.columns(2)
        
        format_dict = {
            'LTP (₹)': '₹{:.2f}', 
            'Chg (₹)': '{:.2f}', 
            '% Chg': '{:.2f}%'
        }

        with col_gainer:
            st.subheader("Top 15 Gainers 🟢")
            st.dataframe(gainers_df_raw.style.format(format_dict), use_container_width=True)
            
        with col_loser:
            st.subheader("Top 15 Losers 🔴")
            st.dataframe(losers_df_raw.style.format(format_dict), use_container_width=True)

    # --- TAB 4: Real-Time News (NEW FEATURE) ---
    with tab4:
        st.header(f"📰 Real-Time News for {display_name}")
        st.info("News is pulled from various sources via yfinance. The list is dynamic and updates periodically.")
        news_data = fetch_stock_news(selected_ticker)
        
        if news_data:
            st.markdown("---")
            for article in news_data[:10]: # Display top 10 articles
                title = article.get('title', 'No Title')
                publisher = article.get('publisher', 'Unknown Source')
                link = article.get('link', '#')
                
                # Convert timestamp to human-readable format
                timestamp = article.get('providerPublishTime')
                if timestamp:
                    dt_object = datetime.fromtimestamp(timestamp)
                    # Simple time ago calculation 
                    now = datetime.now()
                    diff = now - dt_object
                    if diff.total_seconds() < 3600:
                        time_str = f"{int(diff.total_seconds() // 60)} minutes ago"
                    elif diff.total_seconds() < 86400:
                        time_str = f"{int(diff.total_seconds() // 3600)} hours ago"
                    else:
                        time_str = f"{diff.days} days ago"
                else:
                    time_str = 'Date Unknown'
                
                st.markdown(f"""
                <div style="padding: 10px; border-bottom: 1px solid #eee; margin-bottom: 5px;">
                    <a href="{link}" target="_blank" style="font-size: 1.1em; font-weight: 600; color: #1a73e8; text-decoration: none;">{title}</a>
                    <p style="font-size: 0.9em; color: #555; margin: 5px 0 0;">
                        {publisher} | Published: {time_str}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No recent news articles found for {display_name}.")

    # --- TAB 5: Price Prediction (Original tab4 content, moved to tab5) ---
    with tab5:
        st.header(f"{display_name} Simple Price Prediction")
        st.info("⚠️ **Note on Prediction:** This forecast uses a basic Linear Regression model on the time index. **It is not an accurate or reliable tool for real investment decisions.**")

        from sklearn.linear_model import LinearRegression # Re-import for encapsulation, though unnecessary in Streamlit
        
        def simple_prediction_model(data, days_to_predict=7):
            if len(data) < 30:
                return None, "Not enough historical data (minimum 30 data points) for prediction."

            df = data[['Close']].copy()
            df['Time_Index'] = df.index.factorize()[0]
            
            X = df[['Time_Index']].values
            y = df['Close'].values 
            
            model = LinearRegression()
            model.fit(X, y)

            last_index = X[-1][0]
            future_indices = np.array(range(last_index + 1, last_index + 1 + days_to_predict)).reshape(-1, 1)
            
            future_predictions = model.predict(future_indices)
            last_date = df.index[-1].date()
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
            
            prediction_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])
            
            return prediction_df, None
        
        prediction_df, error = simple_prediction_model(data_with_indicators, days_to_predict=7)
        
        if error:
            st.warning(f"Prediction Warning: {error}")
        elif prediction_df is not None:
            col_pred1, col_pred2 = st.columns([1, 2])
            with col_pred1:
                st.subheader("Forecast Table")
                st.dataframe(prediction_df.style.format("₹{:.2f}"), use_container_width=True)
            with col_pred2:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=data_with_indicators.index, y=data_with_indicators['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))
                fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Predicted Close'], mode='lines+markers', name=f'Predicted Close (Next 7 Days)', line=dict(color='orange', dash='dash')))
                fig_pred.update_layout(title='Simple Forecast', xaxis_title='Date', yaxis_title='Price (INR)')
                st.plotly_chart(fig_pred, use_container_width=True)


if __name__ == "__main__":
    main_app()
