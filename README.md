# FOCUS-Backend
Backend code for FOCUS, a master’s group project. This Django-based app uses machine learning to process real-time eye-tracking data, extracting metrics like blinks, fixations, and pupil dilation. It provides APIs for user management, eye data analysis, and session tracking, supporting a frontend for visualising productivity and eye strain.

## Prerequisites
Ensure the following are installed on your system:

1. Python 3.12 (with pip)
2. PostgreSQL (latest version is 17 as of Dec 2024)
3. Virtual environment (venv)
4. Daphne (installed as part of the project dependencies)
5. Git (for version control management)

# Running the Backend Server
Start the backend server using **Uvicorn** (ASGI):

```bash
# Using one single process
uvicorn backend.asgi:application
```

To run with multiple processes and benefit from multi-processing Websockets, use the flag `--workers` and specify the number of processes to launch.

The number of workers you specify should not ideally exceed the number of logical processors you have. For example, for a 4-core 8-thread CPU with 8 logical processors, **8 worker processes** should be used.
Please check the number of CPUs avaialable using Task Manager (Windows), `lscpu` command (Linux environments) or similar in Mac.

```bash
# If 8 logical CPUs are available
uvicorn backend.asgi:application --workers 8
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
   - This Guide will be using Version 17. If you are using another version, make sure the path names contain the correct version number.
2. Follow the installation wizard and set a password for the postgres user.

#### Additional tools for Windows

- Git Bash
   - Included with Git for Windows
- OpenSSL
   - Included with Git for Windows

Please run all of the following commands under a `Git Bash` terminal running as `administrator`.

### 2. Ensure PostgreSQL is running
```bash
brew services start postgresql@14 # macOS

net start postgresql-x64-17 # Windows
```

For Windows, you can check PostgreSQL is running using the pgAdmin GUI or the Windows Services Manager.


### 3. Create a Database and User

Open the PostgreSQL Shell (psql):

#### macOS
```bash 
psql -d postgres # default method for macOS (Homebrew Installation)
psql -U team -d focus # method for macOS if you created a custom user (e.g., team) and database (e.g., focus)
```

#### Windows
1) Open `Git Bash` as **administrator**

2) Go to the directory where the `PostgreSQL\17\data` folder is found, by default it should be in the Program Files folder on your drive.

```bash 
cd C:/ProgramFiles/PostgreSQL/17/data
psql -U postgres # use the default postgres user if you installed PostgreSQL via the PostgreSQL installer
```

You will be prompted to enter the superuser password which you set up when installing PostgreSQL. If you forgot this, you have to reinstall PostgreSQL again.

#### Create a database and user:

```sql
CREATE DATABASE your_database_name;
CREATE USER your_database_user WITH PASSWORD 'your_database_password';
GRANT ALL PRIVILEGES ON DATABASE your_database_name TO your_database_user;
ALTER ROLE your_database_user SET client_encoding TO 'utf8';
ALTER ROLE your_database_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE your_database_user SET timezone TO 'UTC';
ALTER DATABASE your_database_name OWNER TO your_database_user;
```

Exit the shell:
```psql
\q
```

### 4. Create a `.env` File

Inside the first `./Backend` folder of your project (same level as `manage.py`), create a `.env` file:

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

```bash
# macOS
python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'

# Windows
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())" # notice the double quotes, if this does not work, try single quotes
```

Copy and paste the printed value into the `.env` file, enclosed in quotation marks like shown below:

```
SECRET_KEY="your_secret_key"
```

### 6. Check SSL Status

SSL is required for encryption between Django and PostgreSQL. To check whether SSL is enabled for your PostgreSQL setup:

#### 1. Open the PostgreSQL shell:

   ```bash
   psql -U your_database_user -d your_database_name
   ```

   Replace `your_database_user` and `your_database_name` with the credentials and database you created earlier.

#### 2. Run the following SQL command to check the SSL status:

   ```sql
   SHOW ssl;
   ```
   
   If SSL is enabled, the output will show:
   ```psql
   ssl
   -----
   on
   (1 row)
   ```

#### 3. If SSL is not enabled (off), follow [link](https://www.postgresql.org/docs/current/ssl-tcp.html) which is summarised below:

   For Windows, make sure you are running a `Git Bash` terminal as administrator, and you are inside the `\data` subdirectory of the PostgreSQL folder.
   
   - Run the following command in the psql shell to find the path to your postgresql.conf file:

   ```sql
   SHOW config_file; 
   ```

   You may need to login as a superuser if you do not have permission:

   ```sql
   CREATE ROLE postgres WITH SUPERUSER LOGIN PASSWORD 'your_password'; 
   ```

   - Open the postgresql.conf file:

   You can use `nano` text editor like shown below, **or alternatively**, use your preferred text editor (e.g. VScode, Notepad)
   ```bash
   nano /usr/local/var/postgres/postgresql.conf  # Replace with your config path

   # e.g. for Windows, "C:/Program Files/PostgreSQL/17/data/postgresql.conf"
   ```

   - Find and update the following settings:

   *If you are using the `nano` text editor, use Ctrl+W for find tool
   After changes, press Ctrl+O to save, then Ctrl+X to exit*

   ```plaintext
   ssl = on       # Line 107, uncomment and change 'off' to 'on'

   ssl_cert_file = 'server.crt'     # Line 109, uncomment

   ssl_key_file = 'server.key'      # Line 112, uncomment
   ```
   If these lines are commented out (with #), remove the # to uncomment them.

#### 4. Generate SSL Certificates

   If you don’t already have SSL certificates (server.crt and server.key), generate self-signed certificates for local development:

   For Windows, make sure you are running `Git Bash` as admin inside the `\data` subdirectory of the PostgreSQL folder when running these commands.

   This makes sure that the SSL certificates are stored in the right directory and there is no need for file movement.
   
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
   mv server.crt server.key /usr/local/var/postgres/ # macOS

   # Not needed for Windows, since you should already be in the correct directory. If not use:

   mv server.crt server.key /path/to/PostgreSQL/17/data # Change '/path/to' to reflect your directory
   ```

   - Set proper permissions for the private key:
   ```bash
   chmod 600 /usr/local/var/postgres/server.key # macOS

   chmod 600 server.key # Windows
   ```

#### 5. Restart PostgreSQL to apply changes

   ### macOS

   ```bash
   brew services restart postgresql
   ```
   
   ### Windows

   ```bash
   net stop postgresql-x64-17 
   net start postgresql-x64-17
   ```

   If you are unable to start up the server again using `net` command above, then try the following:

   ```bash
   pg_ctl start -D "C:\path\to\PostgreSQL\17\data"  # Update to your path
   ```

#### 6. Verify SSL is Enabled

   - Reconnect to the database using psql (see Step 3) and run:
   ```sql
   SHOW ssl;
   ```
   
   You should see:
   ```psql
   ssl
   -----
   on
   (1 row)
   ```

   - To confirm that a connection is using SSL, run:

   ```psql
   \conninfo
   ```

   You should see something similar to:

   ```bash
   SSL connection (protocol: TLSv1.3, cipher: AES256-GCM-SHA384, bits: 256, compression: off) # macOS

   SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, compression: off, ALPN: postgresql) # Windows
   ```

#### 7. (Optional) Use SSL in Production

   For production environments, replace self-signed certificates with certificates issued by a trusted Certificate Authority (CA), such as Let's Encrypt


#### 8. Open the PostgreSQL Shell using IPv4 SSL conection going forward:

   ```bash 
   psql -U postgres -h localhost
   psql -U team -d focus -h localhost
   ```

### 7. Apply migrations:

Go back to the project directory inside `./Backend` and run the following:

```bash
python manage.py makemigrations
python manage.py migrate
```

If no errors are displayed, and OK messages are shown for each migration, the database is ready and can be used with the website.
