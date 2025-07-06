#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests
import warnings


# In[3]:


# Page configuration
st.set_page_config(
    page_title="Investment Learning Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# In[5]:


# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .news-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# In[7]:


# Initialize LangChain components
def initialize_langchain():
    """Initialize LangChain components for AI assistance"""
    try:
        # You need to set your OpenAI API key
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        if openai_api_key == "OPENAI_API_KEY":
            st.warning("⚠️ Please set your OpenAI API key in Streamlit secrets or environment variables")
            return None
        
        llm = OpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            max_tokens=500
        )
        
        template = """You are an expert investment advisor and financial educator. 
        You help people understand investing concepts, analyze stocks, and make informed decisions.
        
        Current conversation:
        {history}
        
        Human: {input}
        Assistant: """
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        return conversation
    except Exception as e:
        st.error(f"Error initializing AI assistant: {str(e)}")
        return None


# In[11]:


#Function to get stock data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None


# In[13]:


# Function to get market news
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_market_news():
    """Fetch market news (placeholder - replace with actual news API)"""
    # This is a placeholder. You would integrate with a real news API
    sample_news = [
        {
            "title": "Market Opens Higher on Strong Economic Data",
            "summary": "Markets opened with gains as economic indicators show positive trends...",
            "time": "2 hours ago"
        },
        {
            "title": "Tech Stocks Rally on AI Developments",
            "summary": "Technology sector sees significant gains following breakthrough announcements...",
            "time": "4 hours ago"
        },
        {
            "title": "Federal Reserve Policy Update",
            "summary": "Latest monetary policy decisions impact market sentiment...",
            "time": "6 hours ago"
        }
    ]
    return sample_news


# In[15]:


# Function to calculate technical indicators
def calculate_technical_indicators(data):
    """Calculate basic technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data


# In[17]:


# Main application
def main():
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown('<h1 class="main-header">📈 Investment Learning Hub</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("🎯 Navigation")
        
        page = st.selectbox(
            "Choose a section:",
            ["Dashboard", "Stock Analysis", "Learning Center", "Portfolio Tracker", "Market News", "AI Assistant"]
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick stock lookup
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.subheader("🔍 Quick Stock Lookup")
        quick_symbol = st.text_input("Enter stock symbol:", placeholder="e.g., AAPL")
        
        if quick_symbol:
            try:
                stock = yf.Ticker(quick_symbol.upper())
                info = stock.info
                current_price = info.get('currentPrice', 'N/A')
                change = info.get('regularMarketChangePercent', 0)
                
                st.metric(
                    label=f"{quick_symbol.upper()}",
                    value=f"${current_price}",
                    delta=f"{change:.2f}%" if isinstance(change, (int, float)) else "N/A"
                )
            except:
                st.error("Invalid symbol or data unavailable")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content based on page selection
    if page == "Dashboard":
        show_dashboard()
    elif page == "Stock Analysis":
        show_stock_analysis()
    elif page == "Learning Center":
        show_learning_center()
    elif page == "Portfolio Tracker":
        show_portfolio_tracker()
    elif page == "Market News":
        show_market_news()
    elif page == "AI Assistant":
        show_ai_assistant()

def show_dashboard():
    """Display the main dashboard"""
    st.header("📊 Market Dashboard")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("S&P 500", "4,450.25", "1.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("NASDAQ", "13,850.33", "0.8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("DOW", "34,750.12", "0.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("VIX", "18.45", "-2.1%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample chart
    st.subheader("📈 Market Trends")
    
    # Generate sample market data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    prices = np.cumsum(np.random.randn(len(dates)) * 0.01) + 100
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    fig = px.line(df, x='Date', y='Price', title='Market Index Performance')
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_stock_analysis():
    """Display stock analysis tools"""
    st.header("🔍 Stock Analysis")
    
    # Stock selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter stock symbol:", value="AAPL", placeholder="e.g., AAPL, GOOGL, MSFT")
    
    with col2:
        period = st.selectbox("Time period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    if symbol:
        data, info = get_stock_data(symbol.upper(), period)
        
        if data is not None and not data.empty:
            # Stock info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_price = info.get('currentPrice', data['Close'].iloc[-1])
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap = f"${market_cap/1e9:.2f}B"
                st.metric("Market Cap", market_cap)
            
            with col3:
                pe_ratio = info.get('trailingPE', 'N/A')
                if pe_ratio != 'N/A':
                    pe_ratio = f"{pe_ratio:.2f}"
                st.metric("P/E Ratio", pe_ratio)
            
            # Calculate technical indicators
            data_with_indicators = calculate_technical_indicators(data.copy())
            
            # Price chart with indicators
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='green', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{symbol.upper()} Stock Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Chart
                fig_rsi = px.line(
                    x=data_with_indicators.index,
                    y=data_with_indicators['RSI'],
                    title='RSI (Relative Strength Index)'
                )
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # MACD Chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red')
                ))
                fig_macd.update_layout(title='MACD')
                st.plotly_chart(fig_macd, use_container_width=True)
            
            
# Company information
st.subheader("📄 Full Company Information")

if info:
    for key, value in info.items():
        st.write(f"**{key.replace('_', ' ').title()}**: {value}")
else:
    st.info("No additional company information available.")


# Business summary
    if 'longBusinessSummary' in info:
                st.subheader("Business Summary")
                st.write(info['longBusinessSummary'])

def show_learning_center():
    """Display educational content"""
    st.header("📚 Investment Learning Center")
    
    # Learning topics
    topics = {
        "Basics": [
            "What is Stock Market?",
            "Types of Investments",
            "Risk vs Return",
            "Portfolio Diversification"
        ],
        "Technical Analysis": [
            "Chart Patterns",
            "Moving Averages",
            "RSI and MACD",
            "Support and Resistance"
        ],
        "Fundamental Analysis": [
            "Financial Statements",
            "P/E Ratios",
            "DCF Valuation",
            "Industry Analysis"
        ],
        "Advanced Topics": [
            "Options Trading",
            "Derivatives",
            "Algorithmic Trading",
            "Risk Management"
        ]
    }
    
    selected_category = st.selectbox("Choose a learning category:", list(topics.keys()))
    
    st.subheader(f"📖 {selected_category} Topics")
    
    for topic in topics[selected_category]:
        with st.expander(topic):
            st.write(f"Learn about {topic.lower()} and its applications in investment strategy.")
            st.write("This section would contain detailed explanations, examples, and interactive content.")
            
            # Sample content for demonstration
            if topic == "What is Stock Market?":
                st.write("""
                The stock market is a collection of markets where stocks (pieces of ownership in businesses) 
                are traded between investors. It usually refers to the exchanges where stocks and other securities 
                are bought and sold.
                
                **Key Concepts:**
                - Stocks represent ownership in companies
                - Prices fluctuate based on supply and demand
                - Markets provide liquidity for investors
                - Regulation ensures fair trading practices
                """)
            
            # Add interactive quiz or exercise
            st.button(f"Take Quiz on {topic}", key=f"quiz_{topic}")

def show_portfolio_tracker():
    """Display portfolio tracking interface"""
    st.header("💼 Portfolio Tracker")
    
    # Portfolio input
    st.subheader("Add Holdings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL")
    
    with col2:
        shares = st.number_input("Number of Shares", min_value=0.0, step=0.1)
    
    with col3:
        avg_price = st.number_input("Average Price", min_value=0.0, step=0.01)
    
    if st.button("Add to Portfolio"):
        if symbol and shares > 0 and avg_price > 0:
            # Here you would save to a database or session state
            st.success(f"Added {shares} shares of {symbol} at ${avg_price:.2f}")
    
    # Sample portfolio display
    st.subheader("Current Portfolio")
    
    sample_portfolio = pd.DataFrame({
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'Shares': [10, 5, 15, 8],
        'Avg Price': [150.25, 2500.50, 300.75, 3200.00],
        'Current Price': [175.30, 2750.25, 325.50, 3350.75],
        'Value': [1753.00, 13751.25, 4882.50, 26806.00],
        'Gain/Loss': [250.50, 1248.75, 371.25, 1206.00],
        'Gain/Loss %': [16.7, 10.0, 12.4, 4.7]
    })
    
    st.dataframe(sample_portfolio, use_container_width=True)
    
    # Portfolio summary
    total_value = sample_portfolio['Value'].sum()
    total_gain = sample_portfolio['Gain/Loss'].sum()
    total_gain_pct = (total_gain / (total_value - total_gain)) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    
    with col2:
        st.metric("Total Gain/Loss", f"${total_gain:,.2f}")
    
    with col3:
        st.metric("Total Return %", f"{total_gain_pct:.2f}%")
    
    # Portfolio allocation chart
    fig = px.pie(
        sample_portfolio, 
        values='Value', 
        names='Symbol',
        title='Portfolio Allocation'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_market_news():
    """Display market news and updates"""
    st.header("📰 Market News & Updates")
    
    news_data = get_market_news()
    
    for news in news_data:
        st.markdown(f'''
        <div class="news-card">
            <h3>{news["title"]}</h3>
            <p>{news["summary"]}</p>
            <small>📅 {news["time"]}</small>
        </div>
        ''', unsafe_allow_html=True)
    
    # Economic calendar (placeholder)
    st.subheader("📅 Economic Calendar")
    
    calendar_data = pd.DataFrame({
        'Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Event': ['GDP Report', 'Fed Meeting', 'Earnings Season'],
        'Impact': ['High', 'High', 'Medium'],
        'Previous': ['2.1%', '5.25%', 'N/A'],
        'Forecast': ['2.3%', '5.50%', 'N/A']
    })
    
    st.dataframe(calendar_data, use_container_width=True)

def show_ai_assistant():
    """Display AI-powered investment assistant"""
    st.header("🤖 AI Investment Assistant")
    
    # Initialize conversation if not exists
    if st.session_state.conversation is None:
        st.session_state.conversation = initialize_langchain()
    
    if st.session_state.conversation is None:
        st.error("AI Assistant is not available. Please check your API configuration.")
        return
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**AI:** {ai_msg}")
        st.markdown("---")
    
    # Input for new question
    user_input = st.text_input("Ask me anything about investing:", placeholder="e.g., What is a good P/E ratio?")
    
    if st.button("Send") and user_input:
        try:
            # Get AI response
            response = st.session_state.conversation.predict(input=user_input)
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, response))
            
            # Rerun to display new message
            st.rerun()
            
        except Exception as e:
            st.error(f"Error getting AI response: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    
    sample_questions = [
        "What is the difference between value and growth investing?",
        "How do I analyze a company's financial health?",
        "What are the key indicators for market timing?",
        "How should I diversify my portfolio?",
        "What is the impact of inflation on investments?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            # Auto-fill the input
            st.session_state.user_input = question

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




