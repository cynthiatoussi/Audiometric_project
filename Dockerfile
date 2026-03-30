# ==============================
# Dockerfile - Gain Project
# ==============================
#
# Conteneurise le projet Kedro + API Flask.
#
# Construction :
#   docker build -t gain-project .
#
# Exécution (API) :
#   docker run -p 5001:5001 gain-project
#
# Exécution (entraînement dans le container) :
#   docker run gain-project kedro run
#
# ==============================

# Image de base Python

FROM python:3.11-slim

# Répertoire de travail dans le container
WORKDIR /app

# Copier les fichiers de dépendances d'abord (cache Docker)
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Port exposé pour l'API Flask
EXPOSE 5001

# Commande par défaut : lancer l'API
CMD ["python", "app.py"]