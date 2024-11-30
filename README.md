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
brew services start postgresql@14 # macOS

```

For Windows, you can check PostgreSQL is running using the pgAdmin GUI or the Windows Services Manager.


### 3. Create a Database and User

Open the PostgreSQL Shell:

#### macOS
```bash 
psql -d postgres # default method for macOS (Homebrew Installation)
psql -U team -d focus # method for macOS if you created a custom user (e.g., team) and database (e.g., focus)
```

#### Windows
```bash 
psql -U postgres # use the default postgres user if you installed PostgreSQL via the PostgreSQL installer
psql -U postgres -d postgres # you might also need to specify the database
```

#### Create a database and user:
```sql
CREATE DATABASE your_database_name;
CREATE USER your_database_user WITH PASSWORD 'your_database_password';
GRANT ALL PRIVILEGES ON DATABASE your_database_name TO your_database_user;
ALTER ROLE your_database_user SET client_encoding TO 'utf8';
ALTER ROLE your_database_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE your_database_user SET timezone TO 'UTC';
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

### 7. Check SSL Status

SSL is required for encryption between Django and PostgreSQL. To check whether SSL is enabled for your PostgreSQL setup:

1. Open the PostgreSQL shell:
   ```bash
   psql -U your_database_user -d your_database_name
   ```

   Replace your_database_user and your_database_name with the credentials and database you created earlier.

2. Run the following SQL command to check the SSL status:
   ```sql
   SHOW ssl;
   ```
   
   If SSL is enabled, the output will show:
   ```
   ssl
   -----
   on
   (1 row)
   ```

3. If SSL is not enabled (off), follow [link](https://www.postgresql.org/docs/current/ssl-tcp.html)
   
   - Run the following command in the psql shell to find the path to your postgresql.conf file:
   ```sql
   SHOW config_file; 
   ```

   You may need to login as a superuser if you do not have permission:
   ```sql
   CREATE ROLE postgres WITH SUPERUSER LOGIN PASSWORD 'your_password'; 
   ```

   - Open the postgresql.conf file:
   ```bash
   nano /usr/local/var/postgres/postgresql.conf  # Replace with your config path
   ```

   - Find and update the following settings:
   ```plaintext
   ssl = on
   ssl_cert_file = 'server.crt'
   ssl_key_file = 'server.key'
   ```
   If these lines are commented out (with #), remove the # to uncomment them.

4. Generate SSL Certificates

    If you don’t already have SSL certificates (server.crt and server.key), generate self-signed certificates for local development:
   
   - Generate a private key:
   ```bash
   openssl genrsa -out server.key 2048
   ```

   - Generate a certificate signing request (CSR):
   ```bash
   openssl req -new -key server.key -out server.csr
   ```
   When prompted, you can fill in the details or leave them blank.

   - Generate a self-signed certificate:
   ```bash
   openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
   ```

   - Move the certificates to the PostgreSQL directory:
   ```bash
   mv server.crt server.key /usr/local/var/postgres/
   ```

   - Set proper permissions for the private key:
   ```bash
   chmod 600 /usr/local/var/postgres/server.key
   ```

5. Restart PostgreSQL to apply changes
   ```bash
   brew services restart postgresql # Or equivalent
   ```

6. Verify SSL is Enabled
   - Reconnect to the database and run:
   ```sql
   SHOW ssl;
   ```
   
   You should see:
   ```
   ssl
   -----
   on
   (1 row)
   ```

   - To confirm that a connection is using SSL, run:
   ```sql
   \conninfo
   ```

   You should see:
   ```
   SSL connection (protocol: TLSv1.3, cipher: AES256-GCM-SHA384, bits: 256, compression: off)
   ```

7. (Optional) Use SSL in Production

    For production environments, replace self-signed certificates with certificates issued by a trusted Certificate Authority (CA), such as Let's Encrypt


8. Open the PostgreSQL Shell using IPv4 SSL conection going forward:
    ```bash 
    psql -U postgres -h localhost
    psql -U team -d focus -h localhost
    ```

