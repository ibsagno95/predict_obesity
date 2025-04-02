from import_data import load_data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import logging
import pickle

#df = load_data("adeniranstephen/obesity-prediction-dataset",'ObesityDataSet_raw_and_data_sinthetic.csv')
## config logging
log_file = "historique.log"
path_log_file = os.path.join(os.getcwd(), log_file)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                      handlers=[
        logging.FileHandler(path_log_file),  # Stockage dans le fichier
        logging.StreamHandler()  # Affichage sur la console
    ],
      force = True)

def prepare_final_dataset(df):
    ## suppresion de la colonne 'MTRANS'
    if 'MTRANS' in df.columns:
        df = df.drop('MTRANS',axis=1)
    else:
        df=df

    # Sélection des colonnes catégorielles
    col = df.select_dtypes(include=['object', 'category']).columns
    col_one_hot = [x for x in col if x != "NObeyesdad"]
    logging.info(f"Colonnes sélectionnées pour l'encodage One Hot: {col_one_hot}")

    # One Hot Encoder
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    df_col_encoded = one_hot_encoder.fit_transform(df[col_one_hot])

    # Création d'un DataFrame avec les colonnes encodées
    df_col_encoded = pd.DataFrame(df_col_encoded, columns=one_hot_encoder.get_feature_names_out(col_one_hot))

    # Suppression des colonnes catégorielles du DataFrame original
    df_quanti = df.drop(col, axis=1)

    if 'NObeyesdad' in df.columns:
    # Transformation de la variable cible
        vec_obesite = df['NObeyesdad'].unique()
        Val_obesite = range(len(vec_obesite))
        dic_obesite = dict(zip(vec_obesite, Val_obesite))
        df_cible = df['NObeyesdad'].map(dic_obesite)

    # Création du DataFrame final
        df_finale = pd.concat([df_quanti, df_col_encoded, df_cible], axis=1)
    
    else:
        df_finale = pd.concat([df_quanti, df_col_encoded], axis=1)
    
        # Log pour indiquer la réussite de la transformation
    logging.info("Transformation des données réussie. DataFrame final créé.")

    with open('encodage.pkl' , "wb") as enc:
        pickle.dump(one_hot_encoder,enc)
    
    return df_finale



# Appel de la fonction pour charger df depuis import_data.py
 
#prepare_final_dataset(df)


