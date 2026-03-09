import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        connection_timeout=10
    )

views = [
    "vw_buybox_opportunities_summary",
    "vw_buybox_opportunities",
    "vw_buybox_opportunity_amazon_sellers",
    "vw_buybox_opportunity_iw_listings"
]

try:
    cn = get_conn()
    cur = cn.cursor()
    
    for view in views:
        print(f"\n{'='*20} Schema for {view} {'='*20}")
        cur.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{view}'")
        cols = cur.fetchall()
        for col in cols:
            print(f"{col[0]} ({col[1]})")
            
        print(f"\n{'='*20} Sample Data for {view} (Limit 3) {'='*20}")
        cur.execute(f"SELECT * FROM {view} LIMIT 3")
        headers = [d[0] for d in cur.description]
        rows = cur.fetchall()
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(val) for val in row))
        
    cn.close()
except Exception as e:
    print(f"Error: {e}")
