# Base Image
FROM python:3.11.12-slim

# Working directory
WORKDIR /app

# Copy requirements from parent directory
COPY requirements.txt .

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy Python files from ml_app directory
COPY custom_autoscaler/main.py .

# Run the application
CMD ["python", "main.py"]