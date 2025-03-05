# Use official Python image
FROM python:3.11

# Install dependencies for Cmake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
libglib2.0-0

# Set the working directory
WORKDIR /backend

ARG DATABASE_URL
ARG SECRET_KEY
ARG DEBUG
ARG ALLOWED_HOSTS
ARG ENVIRONMENT
ARG FROM_EMAIL
ARG SEND_GRID_API_KEY

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the default port for uvicorn
EXPOSE 8000

# Run database migrations and collect static files
RUN python manage.py migrate
# RUN python manage.py collectstatic --noinput

# Start uvicorn for WebSockets with 8 processes
CMD ["uvicorn", "backend.asgi:application", "--workers", "8"]
