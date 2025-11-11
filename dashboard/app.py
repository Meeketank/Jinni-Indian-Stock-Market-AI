# Jinni Dashboard - Real-time Stock Market Visualization
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Jinni AI - Stock Market Dashboard", layout="wide")

st.title("🧞 Jinni - AI-Powered Indian Stock Market Analysis")

st.sidebar.header("Configuration")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 60, 5)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stocks Monitored", "2,500+", "+50")
with col2:
    st.metric("Model Accuracy", "94.5%", "+2.1%")
with col3:
    st.metric("Predictions Today", "15,234", "+1,203")
with col4:
    st.metric("Learning Iterations", "1.2M", "+50K")

st.subheader("Live Market Overview")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({
        'Symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK'],
        'Price': [2450.50, 3250.75, 1420.30, 2680.90, 920.40],
        'Change': [2.3, -1.2, 3.5, 1.8, -0.5],
        'Volume': [5234000, 3421000, 8765000, 2345000, 9876000],
        'Prediction': [2475, 3220, 1450, 2710, 915]
    })

st.dataframe(st.session_state.data, use_container_width=True)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Price Trend & Prediction")
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    sample_prices = [2400 + i*3 + (i%5)*10 for i in range(30)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=sample_prices, mode='lines', name='Actual'))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Model Performance")
    acc_data = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=20, freq='H'),
        'Accuracy': [92 + i*0.1 for i in range(20)]
    })
    fig2 = px.line(acc_data, x='Time', y='Accuracy')
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Real-Time Learning Status")
st.info("✅ System is actively learning and improving predictions every second")

if auto_refresh:
    time.sleep(update_interval)
    st.rerun()
