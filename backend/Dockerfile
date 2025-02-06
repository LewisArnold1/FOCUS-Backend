# Use official Python image
FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
# Set the working directory
WORKDIR /backend

ARG DATABASE_URL
ARG SECRET_KEY
ARG DEBUG
ARG ALLOWED_HOSTS
ARG ENVIRONMENT

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the default port for Daphne
EXPOSE 8000

# Run database migrations and collect static files
RUN python manage.py migrate
RUN python manage.py collectstatic --noinput

# Start Daphne for WebSockets
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "backend.asgi:application"]
