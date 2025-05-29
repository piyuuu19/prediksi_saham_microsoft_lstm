import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

print("skrip train_model.py mulai dijalankan")

# konfigurasi
TICKER = 'MSFT'
START_DATE_ALL = '2019-05-25'
END_DATE_ALL = '2025-05-26'
TRAIN_END_DATE = '2024-05-25'
TEST_START_DATE = '2024-05-26'
SEQUENCE_LENGTH = 60 # panjang sekuens input untuk prediksi
MODEL_PATH = os.path.join('artifacts', 'best_msft_lstm_model.keras')
SCALER_PATH = os.path.join('artifacts', 'scaler.pkl')

print("konfigurasi selesai")

os.makedirs('artifacts', exist_ok=True) # pastikan folder artifacts ada
print("folder 'artifacts' dicek/dibuat")

# tahap 1: data preparation
# fungsi ini mengunduh, membersihkan, melakukan penskalaan, dan membagi data.
def load_and_prepare_data():
    print("memulai load_and_prepare_data")
    print(f"mengunduh data untuk {TICKER} dari {START_DATE_ALL} hingga {END_DATE_ALL}...")
    try:
        all_data = yf.download(TICKER, start=START_DATE_ALL, end=END_DATE_ALL, progress=False)
        print("selesai yf.download.")
        if all_data is None:
            print("yf.download mengembalikan none.")
            return None, None, None, None, None
        if all_data.empty:
            print(f"tidak ada data yang ditemukan untuk {TICKER} (dataframe kosong).")
            return None, None, None, None, None
        print(f"data ditemukan. jumlah baris awal: {len(all_data)}")
    except Exception as e:
        print(f"error saat mengunduh data: {e}")
        return None, None, None, None, None

    all_data = all_data[['Close']].copy() # hanya gunakan harga penutupan
    all_data.index = pd.to_datetime(all_data.index)

    # bagi data jadi training dan testing
    train_df_raw = all_data[all_data.index <= TRAIN_END_DATE]
    test_df_raw = all_data[all_data.index > TRAIN_END_DATE]
    test_df_raw = test_df_raw[test_df_raw.index >= TEST_START_DATE] # pastikan data tes sesuai tanggal mulai

    print(f"jumlah data training mentah: {len(train_df_raw)}")
    print(f"jumlah data testing mentah: {len(test_df_raw)}")

    if len(train_df_raw) < SEQUENCE_LENGTH + 1:
        print("tidak cukup data untuk training. keluar dari load_and_prepare_data.")
        return None, None, None, None, None

    if len(test_df_raw) < SEQUENCE_LENGTH + 1:
        print("peringatan: tidak cukup data untuk testing. mungkin tidak ada evaluasi.")
        # pass # tidak perlu pass eksplisit di sini jika tidak ada blok else

    scaler = MinMaxScaler(feature_range=(0, 1)) # inisialisasi scaler untuk normalisasi 0-1
    scaled_train_data = scaler.fit_transform(train_df_raw['Close'].values.reshape(-1, 1)) # fit dan transform data training
    print("scaler di-fit pada data training.")

    if not test_df_raw.empty:
        scaled_test_data = scaler.transform(test_df_raw['Close'].values.reshape(-1, 1)) # transform data tes
        print("data tes di-transform.")
    else:
        scaled_test_data = np.array([]) # data tes kosong
        print("tidak ada data tes untuk di-transform.")

    joblib.dump(scaler, SCALER_PATH) # simpan scaler
    print(f"scaler disimpan di {SCALER_PATH}")
    print("selesai load_and_prepare_data.")
    return scaled_train_data, scaled_test_data, scaler, train_df_raw.index, test_df_raw.index

# fungsi ini membuat sekuens data untuk input lstm
def create_sequences(data, sequence_length):
    print(f"membuat sekuens dengan panjang data: {len(data)}")
    xs, ys = [], []
    if len(data) < sequence_length + 1: # cek apakah data cukup untuk membuat minimal 1 sekuens
        print("tidak cukup data untuk membuat sekuens.")
        return np.array(xs), np.array(ys)

    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length), 0]) # input: sekuens harga
        ys.append(data[i + sequence_length, 0]) # output: harga setelah sekuens
    print(f"sekuens dibuat: x shape {np.array(xs).shape}, y shape {np.array(ys).shape}")
    return np.array(xs), np.array(ys)

# tahap 2: model preparation
# fungsi ini membangun arsitektur model lstm
def build_lstm_model(input_shape):
    print("membangun model lstm...")
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.25)) # lapisan dropout untuk regularisasi
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(units=50))) # return_sequences=false (default) untuk layer lstm terakhir sebelum dense
    model.add(Dropout(0.25))
    model.add(Dense(units=25, activation='relu')) # lapisan dense tersembunyi
    model.add(Dense(units=1)) # lapisan output untuk prediksi harga tunggal
    model.compile(optimizer='adam', loss='mean_squared_error') # kompilasi model
    print("model lstm selesai dibangun dan di-compile.")
    return model

def main():
    print("memulai fungsi main")
    scaled_train_data, scaled_test_data, scaler, train_dates_raw, test_dates_raw = load_and_prepare_data()

    if scaled_train_data is None or scaled_train_data.size == 0:
        print("gagal memuat atau mempersiapkan data training. keluar dari main.")
        return

    print("membuat sekuens training")
    X_train, y_train = create_sequences(scaled_train_data, SEQUENCE_LENGTH)

    if scaled_test_data.size > 0: # hanya buat sekuens tes jika ada data tes
        print("membuat sekuens testing")
        X_test, y_test_scaled_actual = create_sequences(scaled_test_data, SEQUENCE_LENGTH)
    else:
        X_test, y_test_scaled_actual = np.array([]), np.array([]) # inisialisasi array kosong jika tidak ada data tes
        print("tidak ada data tes untuk dibuat sekuensnya")


    if X_train.shape[0] == 0: # jika tidak ada sekuens training yang terbentuk
        print("gagal membuat sekuens training. keluar dari main.")
        return

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # reshape input training untuk lstm
    if X_test.shape[0] > 0: # hanya reshape jika ada data tes
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # reshape input tes untuk lstm

    print(f"bentuk x_train: {X_train.shape}, y_train: {y_train.shape}")
    if X_test.shape[0] > 0:
        print(f"bentuk x_test: {X_test.shape}, y_test (scaled): {y_test_scaled_actual.shape}")

    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    print("ringkasan model")
    model.summary() # tampilkan ringkasan arsitektur model

    # tahap 3: model processing
    # melatih model menggunakan data training
    print("\nmemulai pelatihan model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1) # hentikan jika tidak ada peningkatan
    model_checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1) # simpan model terbaik

    history = model.fit(
        X_train, y_train,
        epochs=100, # jumlah maksimum epoch
        batch_size=32,
        validation_split=0.1, # 10% data training untuk validasi
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    print("pelatihan model selesai.")

    # plot loss training dan validasi
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    print("menyimpan gambar loss training")
    plt.savefig(os.path.join('artifacts', 'training_loss.png'))
    print(f"grafik loss training disimpan di artifacts/training_loss.png")

    # tahap 4: testing model
    # mengevaluasi model pada data testing
    if X_test.shape[0] > 0 and y_test_scaled_actual.shape[0] > 0: # pastikan ada data tes untuk evaluasi
        print("\nmengevaluasi model pada data tes...")
        predictions_scaled = model.predict(X_test) # prediksi harga (masih di-scale)
        predictions_actual = scaler.inverse_transform(predictions_scaled) # kembalikan prediksi ke skala asli
        y_test_actual = scaler.inverse_transform(y_test_scaled_actual.reshape(-1, 1)) # kembalikan y_test ke skala asli

        # hitung metrik evaluasi
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        y_test_actual_safe = np.where(y_test_actual == 0, 1e-6, y_test_actual) # hindari pembagian dengan nol untuk mape
        mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual_safe)) * 100

        print(f"\nmetrik evaluasi pada data tes:")
        print(f"  rmse: {rmse:.4f}")
        print(f"  mae:  {mae:.4f}")
        print(f"  mape: {mape:.2f}%")

        # plot perbandingan harga aktual vs prediksi pada data tes
        if len(test_dates_raw) >= SEQUENCE_LENGTH + len(y_test_actual): # pastikan tanggal cukup untuk plot
            start_plot_idx = SEQUENCE_LENGTH
            end_plot_idx = SEQUENCE_LENGTH + len(y_test_actual)
            actual_test_dates = test_dates_raw[start_plot_idx:end_plot_idx]

            if len(actual_test_dates) == len(y_test_actual): # pastikan panjang data dan tanggal cocok
                plt.figure(figsize=(14, 7))
                plt.plot(actual_test_dates, y_test_actual, color='blue', label='Harga Aktual MSFT (Tes)')
                plt.plot(actual_test_dates, predictions_actual, color='red', linestyle='--', label='Prediksi Harga MSFT (Tes)')
                plt.title('Prediksi vs Aktual pada Data Tes')
                plt.xlabel('Tanggal')
                plt.ylabel('Harga Saham MSFT (USD)')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.grid(True)
                print("menyimpan gambar evaluasi tes")
                plt.savefig(os.path.join('artifacts', 'test_predictions_vs_actual.png'))
                print(f"grafik evaluasi disimpan di artifacts/test_predictions_vs_actual.png")
            else:
                print("peringatan: panjang tanggal dan data prediksi/aktual untuk plot tidak cocok.")
        else:
            print("peringatan: tidak cukup tanggal dalam test_dates_raw untuk plot evaluasi.")
    else:
        print("\ntidak ada data tes yang cukup untuk evaluasi model.")

    # tahap 5: deployment ke streamlit
    # (catatan: kode untuk deployment biasanya terpisah dari skrip training ini)

    print("fungsi main selesai")

if __name__ == '__main__':
    print("memanggil main dari __main__")
    main()
    print("skrip train_model.py selesai sepenuhnya")