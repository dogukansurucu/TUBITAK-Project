import streamlit as st
import pandas as pd
import numpy as np
import random
import base64
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import warnings

# Uyarıları görmezden gel
warnings.filterwarnings("ignore")

# Arka plan görselini eklemek için fonksiyon
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        btn_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64.b64encode(file.read()).decode()}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(btn_css, unsafe_allow_html=True)

# Arka plan görselini ayarlama
add_bg_from_local("C:/Users/aysen/Downloads/R.jpeg")

# Rastgele tohumları ayarlama
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds()

# Veriyi yükleme ve ön işleme
df = pd.read_excel("C:/Users/aysen/OneDrive/Masaüstü/rainfall-and-daily-consumption-data-on-istanbul-dams.xlsx")

df = df[['Tarih', 'İstanbul günlük tüketim(m³/gün)']]
df = df.set_index("Tarih")
df.index = pd.to_datetime(df.index)
df['İstanbul günlük tüketim(m³/gün)'] = df['İstanbul günlük tüketim(m³/gün)'] // 100
df['İstanbul günlük tüketim(m³/gün)'] = df['İstanbul günlük tüketim(m³/gün)'].astype(float)
df = np.log(df)

# Eğitim ve test veri setlerini ayırma
train_size = int(len(df) * 0.80)
train, test = df[0:train_size], df[train_size:len(df)]

# Özellikleri oluşturma
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)
train = create_features(train)
test = create_features(test)

FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
TARGET = 'İstanbul günlük tüketim(m³/gün)'

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

# Veriyi normalize etme
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# LSTM için veriyi yeniden şekillendirme
X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Erken durdurma tanımlama
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks=[early_stopping])

# Test verileri üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Tahminleri orijinal ölçeğe döndürme
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1))
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

# Gerçek ve tahmin değerleri birleştirme
results = pd.DataFrame({'Tarih': test.index, 'Gerçek': y_test_orig.flatten(), 'Tahmin': y_pred_orig.flatten()})
results = results.set_index('Tarih')

# Streamlit uygulaması
st.title("İstanbul Günlük Su Tüketim Tahmini")

# Tarih seçimi
selected_date = st.date_input("Tarih Seçin")

# Tahmin butonu
if st.button("Tahmini Göster", key="show_button", help="Tahmin edilen su miktarını gösterir"):
    if selected_date:
        selected_date = pd.to_datetime(selected_date)
        future_features = {'dayofweek': selected_date.dayofweek,
                           'quarter': selected_date.quarter,
                           'month': selected_date.month,
                           'year': selected_date.year,
                           'dayofyear': selected_date.timetuple().tm_yday}

        future_features_normalized = scaler_X.transform([list(future_features.values())])
        future_features_formatted = np.array(future_features_normalized).reshape((1, 1, len(FEATURES)))
        future_prediction = model.predict(future_features_formatted)
        future_prediction_orig = scaler_y.inverse_transform(future_prediction.reshape(-1, 1))
        predicted_value = future_prediction_orig[0][0]

        st.write(f"Tahmin edilen su miktarı {selected_date.strftime('%Y-%m-%d')}: {predicted_value:.2f} m³/gün")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
    background-color: rgba(255, 255, 255, 0.5); /* Burada beyaz renk, %50 şeffaflık ile kullanılmıştır */
    border-radius: 10px; /* Kenarları yuvarlak hale getirir */
    padding: 10px; /* Metin ve sınır arasında boşluk ekler */
    text-align: justify; /* Metni iki yana yaslayarak hizalar */
}
</style>
<div class="big-font">
Tez çalışmamız kapsamında, su talebi tahminlerinin doğruluğunu artırmak amacıyla çeşitli makine öğrenimi modelleri değerlendirilmiş ve karşılaştırılmıştır. 
Bu modellerin performansını inceleyerek, LSTM modelinin su talebi tahminlerinde yüksek doğruluk sağladığı tespit edilmiştir. 
Bu bulgular doğrultusunda, Python programlama dili kullanılarak bir web uygulaması geliştirilmiştir. 
Bu uygulama, kullanıcıların belirli tarihler için geçmiş ve gelecek su talebi tahminlerini yapabilmelerine olanak tanımaktadır. 
Kullanıcı dostu bir arayüze sahip olan uygulama, su yönetimi yetkilileri ve politika yapıcılar için değerli bir kaynak sunmakta, 
su kaynaklarının sürdürülebilir yönetimine katkıda bulunmaktadır.
</div>
""", unsafe_allow_html=True)





