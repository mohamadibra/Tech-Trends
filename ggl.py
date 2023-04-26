import pandas as pd
import streamlit as st
import yfinance as yf

st.write('Google Close Price and Volume')

tickerSymbole = 'GOOGL'

tickerData = yf.Ticker(tickerSymbole)

tickerDF = tickerData.history(period='1d',start='2010-01-01',end='2020-01-01')

st.line_chart(tickerDF.Close)
st.line_chart(tickerDF.Volume)