# FOCUS-Backend
Backend code for FOCUS, a master’s group project. This Django-based app uses machine learning to process real-time eye-tracking data, extracting metrics like blinks, fixations, and pupil dilation. It provides APIs for user management, eye data analysis, and session tracking, supporting a frontend for visualising productivity and eye strain.


## Prerequisites
Ensure the following are installed on your system:

1. Python 3.12 (with pip)
2. PostgreSQL
3. Virtual environment (venv)
4. Daphne (installed as part of the project dependencies)


## Running the Backend Server
Start the backend server using **Daphne** (ASGI):

```
daphne backend.asgi:application
```


## PostgreSQL Setup Guide for Django
Follow this guide to set up PostgreSQL for your Django project.
---

### 1. Install PostgreSQL

#### macOS
```bash
brew install postgresql && brew services start postgresql
```

#### Windows
1. Download and install PostgreSQL from the PostgreSQL Official Website [link](https://www.postgresql.org/download/).
2. Follow the installation wizard and set a password for the postgres user.


### 2. Ensure PostgreSQL is running
```bash
sudo service postgresql start
```

For Windows, you can check PostgreSQL is running using the pgAdmin GUI or the Windows Services Manager.


### 3. Create a Database and User

Open the PostgreSQL Shell:
```bash
psql -U postgres
```

Create a database and user:
```sql
CREATE DATABASE your_database_name;
CREATE USER your_database_user WITH PASSWORD 'your_database_password';
GRANT ALL PRIVILEGES ON DATABASE your_database_name TO your_database_user;
```

Exit the shell:
```
\q
```


### 4. Create a `.env` File

In the root directory of your project (same level as `settings.py`), create a `.env` file:
```bash
touch .env
```


### 5. Add the following variables to the .env file:

```
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432
SECRET_KEY=your_secret_key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1 
```


#### To create your secret key:
```
python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
```


### 6. Apply migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```