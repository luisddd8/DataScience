import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la aplicación
st.title("Predicción de Precios de Viviendas")

# Subtítulo
st.write("Esta aplicación predice el precio de una vivienda basado en sus características.")

# Cargar el modelo y el pipeline de preprocesamiento
model = joblib.load("best_model.pkl")
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")

# Crear campos de entrada para las características
st.sidebar.header("Ingresa las características de la vivienda")

def user_input_features():
    longitude = st.sidebar.number_input("Longitud", value=-122.23)
    latitude = st.sidebar.number_input("Latitud", value=37.88)
    housing_median_age = st.sidebar.number_input("Edad media de la vivienda", value=41)
    total_rooms = st.sidebar.number_input("Total de habitaciones", value=880)
    total_bedrooms = st.sidebar.number_input("Total de dormitorios", value=129)
    population = st.sidebar.number_input("Población", value=322)
    households = st.sidebar.number_input("Hogares", value=126)
    median_income = st.sidebar.number_input("Ingreso medio", value=8.33)
    ocean_proximity = st.sidebar.selectbox("Proximidad al océano", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

    # Crear un DataFrame con los datos ingresados
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity  # Asegúrate de que el nombre de la columna sea correcto
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Obtener los datos ingresados por el usuario
input_df = user_input_features()

# Mostrar los datos ingresados
st.subheader("Datos ingresados")
st.write(input_df)

# Preprocesar los datos usando el pipeline
try:
    input_prepared = preprocessing_pipeline.transform(input_df)
    st.subheader("Datos preprocesados")
    st.write(input_prepared)
except Exception as e:
    st.error(f"Error al preprocesar los datos: {e}")

# Hacer la predicción
if st.button("Predecir"):
    try:
        prediction = model.predict(input_prepared)
        st.subheader("Predicción")
        st.write(f"El precio predicho de la vivienda es: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
