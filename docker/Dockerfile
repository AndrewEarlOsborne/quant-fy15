FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app catalog
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

RUN mkdir -p /app/logs /app/data

ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

EXPOSE 8000

# Logging Healthcheck
HEALTHCHECK --interval=30m --timeout=10s --start-period=5m --retries=3 \
  CMD python -c "import os; exit(0 if os.path.exists('/app/logs/trading_system.log') else 1)"

CMD ["python", "scripts/daily_prediction.py"]