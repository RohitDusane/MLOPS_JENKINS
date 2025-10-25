FROM python:slim-buster

# Set environment variables to prevent writing .pyc files and to keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to take advantage of Docker cache
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Optionally, if your training script requires certain system-level dependencies for specific libraries like XGBoost or LightGBM, you can install them here as well
# For example, LightGBM might require a specific version of cmake or other dependencies.
# RUN apt-get install -y cmake

# Train the model (this will execute the training script inside the container during the build)
RUN python pipeline/training_pipeline.py

# Expose the port (assuming the app runs on port 5000)
EXPOSE 5000

# Start the application
CMD ["python", "application.py"]