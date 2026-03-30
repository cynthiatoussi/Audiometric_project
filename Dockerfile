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

# ==============================
# Dockerfile - Gain Project
# ==============================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Installer le projet en mode dev (pour que gain_project soit importable)
RUN pip install -e .

# Port API Flask
EXPOSE 5001

# Lancer l'API
CMD ["python", "app.py"]