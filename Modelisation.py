from import_data import load_data
from data_prep import prepare_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, recall_score

df = load_data()

df_prep = prepare_final_dataset(df)


## creation de la table train et test

x = df_prep.drop('NObeyesdad',axis=1)
y = df_prep['NObeyesdad']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

### regression logistique 

lr = LogisticRegression()
lr.fit(x_train,y_train)

#evaluation
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average='macro')
recall = recall_score(y_test,y_pred,average='macro')

print(f"l'accuraccy score est {accuracy}")
print(f"le f1 score est {f1}")
print(f"le recall score est {recall}")