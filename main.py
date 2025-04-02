## chargement des packages
import os
import numpy as np
from import_data import load_data
from data_prep import prepare_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, recall_score
from Modelisation import model
import pickle

## chargement des données

df = load_data("adeniranstephen/obesity-prediction-dataset",'ObesityDataSet_raw_and_data_sinthetic.csv')

## préparation des données
#df_prep = prepare_final_dataset(df)

## modélisation
model(df)


def main():
    print('éxécution terminée')

if __name__ == "__main__":
    main()





