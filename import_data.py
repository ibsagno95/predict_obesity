import kagglehub
import pandas as pd
import os
import logging


def load_data(url,file):
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
    path = kagglehub.dataset_download(url)

    print("Path to dataset files:", path)

    file_path = os.path.join(path,file)



## Importation des données 



    try:
        df = pd.read_csv(file_path)
        logging.info(f"Le fichier  a été chargé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier")
    return df


#df = load_data("adeniranstephen/obesity-prediction-dataset",'ObesityDataSet_raw_and_data_sinthetic.csv')





