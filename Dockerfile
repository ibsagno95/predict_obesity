# Utiliser l'image python 

FROM python:3.10.12

## Définir le répertoire de travail 

WORKDIR /app

## Copier les fichiers nécessaires pour l'appli

COPY app_st.py .
COPY encodage.pkl .
COPY logistic_regression_model.pkl .
COPY inference.py .
COPY requirement.txt .

## Installer les dépendances
RUN pip install --no-cache-dir -r requirement.txt
RUN pip install streamlit

#exposer le port utilisé par streamlit
EXPOSE 8501

# Commande pour lancer l'application

CMD [ "streamlit" ,"run", "app_st.py", "--server.port=8501","--server.address=0.0.0.0" ]