import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
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
MODEL_PATH = os.path.join('artifacts', 'best_msft_lstm_model.keras')
SCALER_PATH = os.path.join('artifacts', 'scaler.pkl')

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

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

# Fungsi untuk membuat sekuens data (sama seperti di train_model.py)
def create_sequences(data, sequence_length):
    xs = []
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length), 0])
    return np.array(xs)

# Fungsi untuk memprediksi harga berikutnya
def predict_next_day(model, scaler, last_sequence):
    # Reshape input untuk LSTM: [samples, time steps, features]
    last_sequence_reshaped = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
    # Prediksi harga terakhir (masih dalam skala 0-1)
    predicted_scaled = model.predict(last_sequence_reshaped)
    # Kembalikan ke skala asli
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    return predicted_price

# Fungsi untuk memprediksi beberapa hari ke depan
def predict_future(model, scaler, last_sequence, days=7):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Prediksi hari berikutnya
        current_sequence_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], 1))
        next_pred_scaled = model.predict(current_sequence_reshaped)[0][0]
        
        # Simpan prediksi
        next_pred = scaler.inverse_transform(np.array([[next_pred_scaled]]))[0][0]
        future_predictions.append(next_pred)
        
        # Perbarui sequence dengan menambahkan prediksi baru dan menghapus nilai terlama
        current_sequence = np.append(current_sequence[1:], next_pred_scaled)
    
    return future_predictions

# Header aplikasi
st.title("📈 Prediksi Harga Saham Microsoft (MSFT)")
st.markdown("""
Aplikasi ini menggunakan model LSTM (Long Short-Term Memory) untuk memprediksi harga saham Microsoft (MSFT).
Model telah dilatih menggunakan data historis dari Yahoo Finance.
""")

# Sidebar untuk parameter
st.sidebar.header("Parameter")

# Pilih tanggal
today = datetime.now()
default_start_date = today - timedelta(days=365)  # 1 tahun yang lalu
start_date = st.sidebar.date_input("Tanggal Mulai Data Historis", default_start_date)
end_date = st.sidebar.date_input("Tanggal Akhir Data Historis", today)

# Jumlah hari untuk prediksi ke depan
future_days = st.sidebar.slider("Jumlah Hari Prediksi Ke Depan", 1, 30, 7)

# Load model dan scaler
with st.spinner("Memuat model..."):
    model = load_lstm_model()
    scaler = load_scaler()

if model is None or scaler is None:
    st.error("Model atau scaler tidak dapat dimuat. Pastikan Anda telah menjalankan train_model.py terlebih dahulu.")
    st.stop()

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

# Memproses data untuk prediksi
close_data = data[['Close']].values
scaled_data = scaler.transform(close_data)

# Memastikan kita punya cukup data
if len(scaled_data) <= SEQUENCE_LENGTH:
    st.error(f"Tidak cukup data untuk membuat prediksi. Butuh minimal {SEQUENCE_LENGTH+1} titik data.")
    st.stop()

# Membuat sekuens terakhir untuk prediksi
last_sequence = scaled_data[-SEQUENCE_LENGTH:, 0]

# Prediksi beberapa hari ke depan
with st.spinner("Membuat prediksi..."):
    future_predictions = predict_future(model, scaler, last_sequence, future_days)

# Membuat dataframe untuk hasil prediksi
last_date = data.index[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close': future_predictions
})

# Menampilkan hasil prediksi
st.subheader(f"Prediksi Harga Saham MSFT untuk {future_days} Hari Ke Depan")
col1, col2 = st.columns([3, 1])

with col1:
    # Gabungkan data historis dengan prediksi untuk visualisasi
    hist_data = data[['Close']].copy()
    hist_data.columns = ['Actual Close']
    
    # Membuat grafik dengan Plotly
    fig = go.Figure()
    
    # Data aktual
    fig.add_trace(go.Scatter(
        x=hist_data.index, 
        y=hist_data['Actual Close'],
        mode='lines',
        name='Harga Aktual',
        line=dict(color='blue')
    ))
    
    # Data prediksi
    fig.add_trace(go.Scatter(
        x=future_df['Date'], 
        y=future_df['Predicted Close'],
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='red', dash='dash')
    ))
    
    # Layout grafik
    fig.update_layout(
        title='Harga Aktual vs Prediksi',
        xaxis_title='Tanggal',
        yaxis_title='Harga (USD)',
        hovermode='x unified',
        legend=dict(y=0.99, x=0.01),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(future_df.set_index('Date'), height=300)
    st.caption("Hasil prediksi")

# Menampilkan performa model (jika file gambar tersedia)
st.subheader("Performa Model")
col1, col2 = st.columns(2)

try:
    with col1:
        if os.path.exists(os.path.join('artifacts', 'training_loss.png')):
            img = Image.open(os.path.join('artifacts', 'training_loss.png'))
            st.image(img, caption='Grafik Loss Training dan Validasi')
        else:
            st.info("Grafik loss training tidak tersedia.")
    
    with col2:
        if os.path.exists(os.path.join('artifacts', 'test_predictions_vs_actual.png')):
            img = Image.open(os.path.join('artifacts', 'test_predictions_vs_actual.png'))
            st.image(img, caption='Evaluasi Model pada Data Testing')
        else:
            st.info("Grafik evaluasi model tidak tersedia.")
except Exception as e:
    st.error(f"Error menampilkan gambar: {e}")

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
    
    © 2025 Prediksi Saham Microsoft LSTM
    """
)