FROM python:3.9-slim-bullseye

# Set environment variables to prevent writing .pyc files and to keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app/MLOPS_Project_1
ENV PYTHONPATH=/app/MLOPS_Project_1/src:$PYTHONPATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . .

# Install the package requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Train the model before running the application
RUN python pipeline/training_pipeline.py

# Expose the port (assuming the app runs on port 5000)
EXPOSE 5000

# Start the application
CMD ["python", "application.py"]
