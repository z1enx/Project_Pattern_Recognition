import os
import pandas as pd
import numpy as np
import warnings
import time
import joblib

warnings.filterwarnings('ignore')

# Konfigurasi konstanta
MODEL_FILE = 'model_rf_murni.joblib'
DATA_FILE = 'dataset.csv'

def bersihkan_harga(nilai):
    if pd.isna(nilai) or nilai == '-': 
        return np.nan
    try:
        val = float(str(nilai).replace('Rp', '').replace(',', '').strip())
        return val if val > 0 else np.nan
    except: 
        return np.nan

def muat_dan_bersihkan_data():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] File '{DATA_FILE}' tidak ditemukan di folder ini!")
        return None
        
    df = pd.read_csv(DATA_FILE)
    df['Nama Provinsi'] = df['Nama Provinsi'].astype(str).str.strip()
    df['Komoditas'] = df['Komoditas'].astype(str).str.strip()
    df['Harga_Numerik'] = df['Harga'].apply(bersihkan_harga)
    
    # Mapping Bulan
    bulan_map = {
        'Januari':1, 'Februari':2, 'Maret':3, 'April':4, 'Mei':5, 'Juni':6,
        'Juli':7, 'Agustus':8, 'September':9, 'Oktober':10, 'November':11, 'Desember':12
    }
    df['Bulan_Angka'] = df['Bulan'].astype(str).str.strip().map(bulan_map)
    
    # Isi missing values & urutkan
    df = df.sort_values(['Nama Provinsi', 'Komoditas', 'Tahun', 'Bulan_Angka']).reset_index(drop=True)
    df['Harga_Numerik'] = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga_Numerik'].ffill(limit=3)
    
    # Sesuai training, kita isi sisa NaN dengan mean
    group_mean = df.groupby(['Nama Provinsi', 'Komoditas'])['Harga_Numerik'].transform('mean')
    global_mean = df.groupby('Komoditas')['Harga_Numerik'].transform('mean')
    df['Harga_Numerik'] = df['Harga_Numerik'].fillna(group_mean).fillna(global_mean).fillna(0.0)
    
    return df

def dapatkan_daftar_pilihan(features):
    # Mengambil daftar Provinsi dari kolom dummy
    provinces = [col.replace('Nama Provinsi_', '') for col in features if col.startswith('Nama Provinsi_')]
    provinces.append('Aceh')  # Baseline
    provinces.sort()
    
    # Mengambil daftar Komoditas dari kolom dummy
    commodities = [col.replace('Komoditas_', '') for col in features if col.startswith('Komoditas_')]
    commodities.append('Bawang Merah')  # Baseline
    commodities.sort()
    
    return provinces, commodities

def prepare_feature_vector(province, commodity, input_price, df, feature_columns):
    # Filter data historis
    df_slice = df[(df['Nama Provinsi'] == province) & (df['Komoditas'] == commodity)].copy()
    if df_slice.empty:
        raise ValueError(f"Tidak ada data historis untuk {commodity} di provinsi {province}.")
        
    df_slice = df_slice.sort_values(by=['Tahun', 'Bulan_Angka']).reset_index(drop=True)
    
    # Ambil baris terakhir di dataset
    last_row = df_slice.iloc[-1]
    last_year = int(last_row['Tahun'])
    last_month = int(last_row['Bulan_Angka'])
    
    # Bulan target prediksi (bulan berikutnya)
    pred_month = last_month + 1
    pred_year = last_year
    if pred_month > 12:
        pred_month = 1
        pred_year += 1
        
    # Susun Lag_1 sampai Lag_12
    # Lag_1 = input_price (harga bulan terakhir dari dosen)
    # Lag_2 = harga dari baris terakhir di DB, dst.
    lags = [input_price]
    for i in range(11):
        # Mulai dari len - 2 karena data di len - 1 (bulan terakhir) digantikan oleh input_price
        idx = len(df_slice) - 2 - i
        if idx >= 0:
            lags.append(df_slice.iloc[idx]['Harga_Numerik'])
        else:
            # Fallback jika data historis kurang dari 12 bulan
            lags.append(input_price)
            
    # Hitung Rolling features dari Lag_1, Lag_2, Lag_3
    rolling_prices = lags[:3]
    rolling_mean = np.mean(rolling_prices)
    rolling_std = np.std(rolling_prices) if len(rolling_prices) > 1 else 0.0
    
    # Buat dictionary fitur numerik
    features_dict = {
        'Tahun': pred_year,
        'Bulan_Sin': np.sin(2 * np.pi * pred_month / 12),
        'Bulan_Cos': np.cos(2 * np.pi * pred_month / 12),
        'Ramadan_Lebaran': 1 if pred_month in [3, 4, 5] else 0,
        'Rolling_Mean_3': rolling_mean,
        'Rolling_Std_3': rolling_std
    }
    
    for i, lag_val in enumerate(lags, 1):
        features_dict[f'Lag_{i}'] = lag_val
        
    # Susun baris input sesuai dengan kolom features model
    row_data = {col: 0.0 for col in feature_columns}
    
    # Masukkan fitur numerik
    for col, val in features_dict.items():
        if col in row_data:
            row_data[col] = val
            
    # Masukkan dummy variables (One-Hot Encoding)
    prov_col = f"Nama Provinsi_{province}"
    if prov_col in row_data:
        row_data[prov_col] = 1.0
        
    kom_col = f"Komoditas_{commodity}"
    if kom_col in row_data:
        row_data[kom_col] = 1.0
        
    df_features = pd.DataFrame([row_data], columns=feature_columns)
    return df_features, pred_year, pred_month

def jalankan_simulasi():
    print("\n" + "="*70)
    print("      SIMULASI PREDIKSI HARGA PANGAN (PRE-TRAINED RANDOM FOREST)      ")
    print("="*70)
    
    # 1. Load Model
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] File model '{MODEL_FILE}' tidak ditemukan!")
        print("Silakan jalankan script training terlebih dahulu: python BELAJAR/2_RF_Murni.py")
        return
        
    print("[SYSTEM] Memuat model pra-latih (Random Forest Murni)...")
    model_data = joblib.load(MODEL_FILE)
    model = model_data['model']
    features = model_data['features']
    
    # 2. Load Data
    print("[SYSTEM] Memuat data historis dari dataset.csv...")
    df = muat_dan_bersihkan_data()
    if df is None:
        return
        
    provinces, commodities = dapatkan_daftar_pilihan(features)
    
    # Default Provinsi
    provinsi_aktif = "Nusa Tenggara Barat"
    if provinsi_aktif not in provinces:
        provinsi_aktif = provinces[0]
        
    print("[SYSTEM] Sistem siap. Menampilkan menu antarmuka.\n")
    time.sleep(0.5)

    while True:
        # Tampilkan Status Provinsi Aktif
        print(f"PROVINSI AKTIF SAAT INI: \033[1;32m{provinsi_aktif}\033[0m")
        print("Silakan pilih komoditas di bawah ini:")
        print("-" * 40)
        
        for i, kom in enumerate(commodities, 1):
            print(f" {i:2d}. {kom}")
        print("  P. Ubah Provinsi Aktif")
        print("  0. Keluar dari Simulasi")
        print("-" * 40)
        
        pilihan = input("[INPUT] Masukkan pilihan Anda: ").strip()
        
        if pilihan == '0':
            print("\nProgram simulasi dihentikan. Sampai jumpa!\n")
            break
            
        if pilihan.upper() == 'P':
            print("\nDaftar Provinsi:")
            for p_idx, p_name in enumerate(provinces, 1):
                print(f" {p_idx:2d}. {p_name}")
            p_pilihan = input("\n[INPUT] Pilih nomor provinsi baru: ").strip()
            try:
                p_num = int(p_pilihan) - 1
                if 0 <= p_num < len(provinces):
                    provinsi_aktif = provinces[p_num]
                    print(f"\n[SYSTEM] Provinsi berhasil diubah menjadi: {provinsi_aktif}\n")
                else:
                    print("\n[WARNING] Nomor provinsi tidak terdaftar.\n")
            except ValueError:
                print("\n[WARNING] Harap masukkan angka bulat.\n")
            continue
            
        try:
            idx = int(pilihan) - 1
            if idx < 0 or idx >= len(commodities):
                print("[WARNING] Nomor komoditas tidak valid.\n")
                continue
        except ValueError:
            print("[WARNING] Masukkan angka bulat atau 'P' / '0'.\n")
            continue
            
        komoditas_terpilih = commodities[idx]
        print(f"\n>> Komoditas: {komoditas_terpilih}")
        
        # Cari data historis bulan terakhir untuk mendapatkan informasi tanggal & harga
        df_slice = df[(df['Nama Provinsi'] == provinsi_aktif) & (df['Komoditas'] == komoditas_terpilih)]
        if df_slice.empty:
            print(f"[ERROR] Data historis {komoditas_terpilih} tidak tersedia untuk provinsi {provinsi_aktif}.\n")
            continue
            
        df_slice_sorted = df_slice.sort_values(by=['Tahun', 'Bulan_Angka'])
        last_row = df_slice_sorted.iloc[-1]
        last_month_name = last_row['Bulan']
        last_year = int(last_row['Tahun'])
        last_price = last_row['Harga_Numerik']
        
        print(f"[INFO] Data historis terakhir tercatat pada: {last_month_name} {last_year}")
        print(f"[INFO] Harga historis terakhir di database: Rp {last_price:,.0f}")
        
        # Minta input harga saat ini
        harga_input = input(f"[INPUT] Masukkan harga {komoditas_terpilih} untuk {last_month_name} {last_year}: ").strip()
        try:
            # Bersihkan jika dosen mengetik Rp, titik, atau koma
            harga_sekarang = float(harga_input.replace('Rp', '').replace('.', '').replace(',', '').strip())
        except ValueError:
            print("[WARNING] Input harga salah. Batal memprediksi.\n")
            continue
            
        # 3. Prediksi menggunakan model asli
        try:
            X_input, pred_year, pred_month = prepare_feature_vector(
                provinsi_aktif, komoditas_terpilih, harga_sekarang, df, features
            )
            
            print(f"\n[SYSTEM] Mengambil riwayat harga masalalu secara otomatis...")
            time.sleep(0.5)
            print(f"[SYSTEM] Mengumpankan fitur ke model Random Forest Murni (Tanpa Leakage)...")
            
            # Prediksi
            pred = model.predict(X_input)
            harga_prediksi = pred[0]
            
            bulan_names = {
                1:'Januari', 2:'Februari', 3:'Maret', 4:'April', 5:'Mei', 6:'Juni',
                7:'Juli', 8:'Agustus', 9:'September', 10:'Oktober', 11:'November', 12:'Desember'
            }
            pred_month_name = bulan_names[pred_month]
            
            # Tampilkan Hasil
            print("\n" + "="*60)
            print(f"       HASIL PREDIKSI UNTUK BULAN DEPAN ({pred_month_name.upper()} {pred_year})")
            print("="*60)
            print(f" Provinsi             : {provinsi_aktif}")
            print(f" Komoditas            : {komoditas_terpilih}")
            print(f" Harga Bulan Ini ({last_month_name})  : Rp {harga_sekarang:,.0f}")
            print(f" Prediksi Bulan Depan ({pred_month_name}): Rp {harga_prediksi:,.0f}")
            
            selisih = abs(harga_prediksi - harga_sekarang)
            if harga_prediksi > harga_sekarang:
                print(f" Tren Pergerakan      : 📈 NAIK (+Rp {selisih:,.0f})")
            elif harga_prediksi < harga_sekarang:
                print(f" Tren Pergerakan      : 📉 TURUN (-Rp {selisih:,.0f})")
            else:
                print(f" Tren Pergerakan      : ➖ STABIL")
            print("-" * 60)
            print("[CATATAN] Model dilatih secara kronologis bebas kebocoran data (2021-2024).")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan dalam pemrosesan prediksi: {str(e)}\n")
            
        input("Tekan Enter untuk melanjutkan...")
        print("\n" * 2)

if __name__ == "__main__":
    jalankan_simulasi()
