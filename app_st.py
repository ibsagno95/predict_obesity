import streamlit as st
import pandas as pd
import numpy as np
import pickle
from inference import prediction

# Paramètre couleur de fond

def set_background():
    st.markdown(
        """
        <style>
            /* Changer la couleur de fond de la sidebar */
            [data-testid="stSidebar"] {
                background-color: #d3d3d3; /* Gris */
            }
            /* Changer la couleur de fond du panneau principal */
            [data-testid="stAppViewContainer"] {
                background-color: #add8e6; /* Bleu clair */
            }
        </style>
        """,
        unsafe_allow_html=True
    )



## Titre de l'application

set_background()

st.title("Prédition de l'obesité ")

with st.sidebar:
    st.title("Informations utilisateur")
    st.header("Saisie des données")
    gender = st.selectbox("Genre", ["Male", "Female"])
    age = st.number_input("Âge", min_value=0, max_value=120, value=30)
    height = st.number_input("Taille (m)", min_value=0.5, max_value=2.5, value=1.65)
    weight = st.number_input("Poids (kg)", min_value=10, max_value=200, value=75)
    family_history = st.selectbox("Antécédents familiaux d'obésité", ["yes", "no"])
    favc = st.selectbox("Consommation fréquente d'aliments hypercaloriques (FAVC)", ["yes", "no"])
    fcvc = st.slider("Consommation de légumes (FCVC)", min_value=0.0, max_value=5.0, value=5.0)
    ncp = st.slider("Nombre de repas principaux par jour (NCP)", min_value=1.0, max_value=5.0, value=2.0)
    caec = st.selectbox("Consommation d'aliments entre les repas (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
    smoke = st.selectbox("Fumez-vous ?", ["yes", "no"])
    ch2o = st.slider("Consommation d'eau quotidienne (CH2O en litres)", min_value=0.0, max_value=5.0, value=1.0)
    scc = st.selectbox("Suivez-vous un régime alimentaire ? (SCC)", ["yes", "no"])
    faf = st.slider("Fréquence d'activité physique par semaine (FAF)", min_value=0.0, max_value=7.0, value=3.0)
    tue = st.slider("Temps d'utilisation des écrans par jour (TUE en heures)", min_value=0.0, max_value=10.0, value=1.0)
    calc = st.selectbox("Consommation d'alcool (CALC)", ["Sometimes", "Frequently", "Always", "no"])
        
user_data = {
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history_with_overweight": family_history,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "CALC": calc
}


if st.button ('afficher les données'):
    df = pd.DataFrame([user_data])
    st.dataframe(df)

if st.button("Faire une prédition"):
    pred = prediction(user_data)
    st.text(f"{pred}")
