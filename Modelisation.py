import os
import numpy as np
from import_data import load_data
from data_prep import prepare_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, recall_score
import pickle
import logging

#df = load_data("adeniranstephen/obesity-prediction-dataset",'ObesityDataSet_raw_and_data_sinthetic.csv')

# df_prep = prepare_final_dataset(df)

def model(df,save_path="logistic_regression_model.pkl"):

    ## config logging
    log_file = "historique.log"
    path_log_file = os.path.join(os.getcwd(), log_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                      handlers=[
        logging.FileHandler(path_log_file),  # Stockage dans le fichier
        logging.StreamHandler()  # Affichage sur la console
    ],
      force = True)
    
## préparation des données pour la modélisation 
    df_prep = prepare_final_dataset(df)

## creation de la table train et test

    x = df_prep.drop('NObeyesdad',axis=1)
    y = df_prep['NObeyesdad']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

### regression logistique 

## selection des variables 

    lr = LogisticRegression(max_iter=1000)
    # sfm = SelectFromModel(lr,threshold = 'mean')
    # sfm.fit(x,y)
    # selected_features = x.columns[sfm.get_support()].tolist()

## entrainement du modèle

    lr.fit(x_train,y_train)


#evaluation
    y_pred = lr.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average='macro')
    recall = recall_score(y_test,y_pred,average='macro')
    y_prob = lr.predict_proba(x_test)


    try: 
       # Sauvegarde du modèle
        with open(save_path, "wb") as f:
            pickle.dump(lr, f)
        logging.info(f"modele enregistré avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de l'entranement du modèle")
        

    return {"model":lr, "features":x.columns,'accuracy':accuracy ,
             "accuracy":accuracy , "probabilité":y_prob,
             "recall": recall , "f1":f1}


