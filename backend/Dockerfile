# Use official Python image
FROM python:3.11

# Set the working directory
WORKDIR /backend

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
