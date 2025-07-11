# 📈 Prediksi Saham Microsoft (MSFT) Menggunakan LSTM

Proyek ini bertujuan untuk memprediksi harga saham Microsoft (MSFT) menggunakan model **Long Short-Term Memory (LSTM)**, sebuah arsitektur *Recurrent Neural Network* (RNN) yang sangat efektif dalam menangani dan memprediksi data deret waktu (*time series*).

---

## 📁 Struktur Proyek

Berikut adalah struktur direktori dari proyek ini:

├── artifacts

├── app.py 

├── train_model.py

├── requirements.txt 

├── runtime.txt 

├── .gitignore 

└── README.md 

---

## 🔍 Tujuan Proyek

- Mengumpulkan data historis saham Microsoft (`MSFT`) dari **Yahoo Finance** menggunakan library `yfinance`.
- Melakukan pra-pemrosesan data, termasuk normalisasi dan pembuatan sekuens data untuk model LSTM.
- Membangun dan melatih model LSTM untuk memprediksi harga penutupan (`Close`) saham.
- Mengevaluasi performa model untuk mengukur akurasinya.
- Membuat aplikasi web interaktif menggunakan **Streamlit** untuk memvisualisasikan data dan hasil prediksi secara dinamis.

---

## 📊 Dataset

- **Sumber Data:** Yahoo Finance (`yfinance`)
- **Kode Saham:** MSFT (Microsoft Corp.)
- **Periode Data:** 25 Mei 2019 – 25 Mei 2025
- **Fitur Utama:** `Open`, `High`, `Low`, `Close`, `Volume`
- **Target Prediksi:** `Close` (Harga Penutupan)

---

## 🧠 Arsitektur Model

Model prediksi dibangun menggunakan arsitektur jaringan LSTM dengan detail sebagai berikut:
- **Model:** Long Short-Term Memory (LSTM)
- **Arsitektur Jaringan:**
  - 2 Layer LSTM
  - 1 Layer Dense (sebagai output)
- **Optimizer:** `Adam`
- **Loss Function:** `Mean Squared Error` (MSE)

---
