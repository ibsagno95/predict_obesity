import pickle
import pandas as pd




feature = {'Gender':'Male', 'Age':30 ,'Height':1.65, 'Weight':75, 
           'family_history_with_overweight':'no',
       'FAVC':'yes', 'FCVC':5.0, 'NCP':2.0, 'CAEC':'Sometimes', 
       'SMOKE':'no', 'CH2O':1.0, 'SCC':'yes', 'FAF':3.0, 'TUE':1.0,
       'CALC':'Frequently'

}


def prediction(dict):

    df_test = pd.DataFrame([dict])

    # Sélection des colonnes catégorielles
    col = df_test.select_dtypes(include=['object', 'category']).columns
    
    #selection des variables quanti
    df_test_quanti = df_test.drop(col,axis=1)

    # chargement de l'encodage

    with open("encodage.pkl" ,"rb") as enc:
       encoder =  pickle.load(enc)
    
    #application de l'encodage
    df_test_class = pd.DataFrame(encoder.transform(df_test[col]), 
                                 columns=encoder.get_feature_names_out(col))


    ## df_test finale

    df_test_finale = pd.concat([df_test_quanti,df_test_class],axis=1)

    ## chargement du modèle

    with open("logistic_regression_model.pkl","rb") as lr:
        model = pickle.load(lr)

    prediction = model.predict(df_test_finale)[0]

    predict_proba = max(model.predict_proba(df_test_finale)[0])

    # Dictionnaire des catégories d'obésité
    obesity_dict = {
    0: 'Normal_Weight',
    1: 'Overweight_Level_I',
    2: 'Overweight_Level_II',
    3: 'Obesity_Type_I',
    4: 'Insufficient_Weight',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
    }

    return f"la prédiction est: **{obesity_dict[prediction]}** avec une probalité de **{predict_proba:.2%}**"






