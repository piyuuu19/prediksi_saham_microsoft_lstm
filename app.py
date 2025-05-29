import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Saham Microsoft",
    page_icon="📈",
    layout="wide"
)

# Konstanta
TICKER = 'MSFT'
SEQUENCE_LENGTH = 60  # Sesuai dengan yang digunakan pada model training

# Header aplikasi
st.title("📈 Prediksi Harga Saham Microsoft (MSFT)")
st.markdown("""
Aplikasi ini menggunakan model LSTM (Long Short-Term Memory) untuk memprediksi harga saham Microsoft (MSFT).
Model telah dilatih menggunakan data historis dari Yahoo Finance.

**CATATAN: Ini adalah versi demo dari aplikasi. Untuk versi lengkap dengan prediksi real-time, aplikasi perlu dijalankan di lingkungan lokal.**
""")

# Sidebar untuk parameter
st.sidebar.header("Parameter")

# Pilih tanggal
today = datetime.now()
default_start_date = today - timedelta(days=365)  # 1 tahun yang lalu
start_date = st.sidebar.date_input("Tanggal Mulai Data Historis", default_start_date)
end_date = st.sidebar.date_input("Tanggal Akhir Data Historis", today)

# Fungsi untuk mengunduh data saham
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"Tidak ada data yang ditemukan untuk {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error saat mengunduh data: {e}")
        return None

# Mengunduh data saham
with st.spinner("Mengunduh data saham terbaru..."):
    data = fetch_stock_data(TICKER, start_date, end_date)

if data is None or data.empty:
    st.error("Tidak dapat mengunduh data saham. Periksa koneksi internet Anda.")
    st.stop()

# Menampilkan data historis
st.subheader("Data Historis Harga Saham Microsoft")
col1, col2 = st.columns([3, 1])

with col1:
    # Plot data harga penutupan dengan Plotly
    fig = px.line(data, x=data.index, y='Close', title='Harga Penutupan Saham MSFT')
    fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga (USD)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10), height=300)
    st.caption("10 data terakhir")

# Informasi tambahan
st.subheader("Tentang Model")
st.markdown("""
Model LSTM (Long Short-Term Memory) digunakan untuk prediksi ini dengan konfigurasi:
- Sequence Length: 60 hari
- 3 layer LSTM Bidirectional
- Dropout layers untuk menghindari overfitting
- Model dilatih menggunakan Mean Squared Error (MSE)

**Catatan Penting:**
- Prediksi pasar saham memiliki ketidakpastian tinggi dan dipengaruhi banyak faktor
- Harap gunakan prediksi ini sebagai salah satu referensi saja, bukan keputusan investasi utama
- Model hanya mempertimbangkan data historis harga, bukan berita atau fundamental
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Tentang")
st.sidebar.info(
    """
    Aplikasi ini dibuat menggunakan:
    - Streamlit
    - TensorFlow/Keras
    - Pandas
    - yfinance
    - Plotly
    
    Untuk mengakses versi lengkap dengan prediksi, jalankan aplikasi di lingkungan lokal.
    """
)