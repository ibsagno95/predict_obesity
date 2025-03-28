import kagglehub
import pandas as pd
import os
import logging

## config logging
log_file = "historique.log"
path_log_file = os.path.join(os.getcwd(), log_file)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                      handlers=[
        logging.FileHandler(path_log_file),  # Stockage dans le fichier
        logging.StreamHandler()  # Affichage sur la console
    ],
      force = True)

# Download latest version
path = kagglehub.dataset_download("adeniranstephen/obesity-prediction-dataset")

print("Path to dataset files:", path)

file_path = os.path.join(path,'ObesityDataSet_raw_and_data_sinthetic.csv')



## Importation des données 

def load_data():

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Le fichier  a été chargé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier")
    return df

#df = load_data()





