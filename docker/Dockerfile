FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API, training script, and saved model
COPY app.py .
COPY bank_mlops_pipeline.py .
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
