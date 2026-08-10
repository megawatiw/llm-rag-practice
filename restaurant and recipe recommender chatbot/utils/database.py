import psycopg2
from psycopg2 import sql
from pathlib import Path

db_config = {
    'host': 'localhost',
    'database': 'food-and-restaurants',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

class DatabaseConnection:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists using information_schema"""
        self.cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """, (table_name,))
        return self.cursor.fetchone()[0]


    # ===== CREATE =====
    def create_table(self, query):
        """Create restaurants table"""
        self.cursor.execute(query)
        self.conn.commit()


    # ===== CREATE =====
    def insert_item(self, query, values):
        """Insert item into table"""
        self.cursor.execute(query, values)
        self.conn.commit()


def get_chromadb_dir():
    return str((Path.home() / "chroma_multimodal").resolve())
