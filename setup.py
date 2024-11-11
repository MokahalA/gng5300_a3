import os
import shutil
import sqlite3
import pandas as pd

'''
    This is a script to setup the SQLite database files for the skincare products & shopping carts.
    If setup.py is run again, it will reset the local database to its original state.
'''

# File paths
local_file = "skincare.sqlite"
backup_file = "skincare.backup.sqlite"
csv_file = "skincare_products.csv"
overwrite = True

# Create the SQLite database file (it if not does not exist)
if overwrite or not os.path.exists(local_file):
    with open(local_file, "wb") as f:
        pass 

# Backup - we use this to "reset" the DB whenever setup.py is run again.
shutil.copy(local_file, backup_file)

# Connect to the SQLite database
conn = sqlite3.connect(local_file)
cursor = conn.cursor()

# Create the "products" table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        description TEXT,
        category TEXT,
        stock INTEGER,
        price REAL
    )
""")

# Create the "shopping_carts" table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS shopping_carts (
        user_id INTEGER,
        product_id INTEGER,
        product_name TEXT NOT NULL,
        price REAL,
        quantity INTEGER,
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
""")

# Load data from the CSV file and insert it into the "products" table
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    
    # Insert data into the "products" table
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO products (product_name, description, category, stock, price)
            VALUES (?, ?, ?, ?, ?)
        """, (row['product_name'], row['description'], row['category'], row['stock'], row['price']))
    
    print("Products table populated successfully.")
else:
    print(f"CSV file '{csv_file}' not found.")

# Commit changes
conn.commit()
# Backup the database file
shutil.copy(local_file, backup_file)
conn.close()

print("Database setup complete!")
