# Base Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy serialized model, application code, and test script
COPY model.bin .
COPY predict.py .
COPY test.py .

# Expose FastAPI application port
EXPOSE 8000

# Start Uvicorn server serving predict.py
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
