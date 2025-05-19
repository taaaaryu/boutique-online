FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy operator code
COPY k8s_operator/ /app/k8s_operator/
COPY config/ /app/config/

# Environment variables
ENV PYTHONPATH=/app
ENV WATCH_NAMESPACE=boutique
ENV OPERATOR_NAME=appconfig-operator

ENTRYPOINT ["python", "-m", "k8s_operator.appconfig_operator"]
