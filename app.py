import streamlit as st
import pickle
import pandas as pd
from pycaret.classification import *

# Load model dari file .sav
@st.cache_resource
def load_model():
    with open('best_model.sav', 'rb') as f:
        model = pickle.load(f)
    return model

# Mengambil input gejala pengguna dari Streamlit
def get_user_input(symptoms):
    user_input = []
    for symptom in symptoms:
        user_input.append(st.slider(f"Severity for {symptom}", 0, 3, 2))  # Anggap severity bisa antara 0-3
    return user_input

# Load model
model = load_model()

# Streamlit UI
st.title("Sistem Diagnosis Penyakit")

# Ambil data gejala (Misal data contoh)
symptoms = ['Symptom_1', 'Symptom_2', 'Symptom_3']  # Ganti dengan nama gejala yang sesungguhnya
user_input = get_user_input(symptoms)

# Mengubah input pengguna menjadi format yang bisa dipakai oleh model
user_data = pd.DataFrame([user_input], columns=symptoms)

# Melakukan prediksi
if st.button("Diagnose"):
    result = predict_model(model, data=user_data)
    disease = result['Label'].iloc[0]  # Ganti 'Label' sesuai dengan nama kolom prediksi dari model

    st.write(f"Hasil Diagnosis: {disease}")
