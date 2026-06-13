FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema requeridas por pandas/xgboost si es necesario
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Usar un volumen persistente si está disponible
# VOLUME /app/data

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
