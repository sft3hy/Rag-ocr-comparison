# src/utils/db_utils.py

import sqlite3
import json
import os
from datetime import datetime


class DatabaseManager:
    """Handles all SQLite database operations for storing and retrieving session data."""

    def __init__(self, db_path="data/history.db"):
        """
        Initializes the database manager and ensures the database and tables exist.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        # Ensure the directory for the database exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = self._create_connection()
        self._initialize_database()

    def _create_connection(self):
        """Creates and returns a database connection."""
        try:
            return sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def _initialize_database(self):
        """Creates the necessary tables if they don't already exist."""
        if not self.conn:
            return

        create_docs_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            vision_model_used TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            chart_dir TEXT,
            faiss_index_path TEXT,
            chunks_path TEXT,
            chart_descriptions_json TEXT
        );
        """

        create_queries_table_sql = """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            sources_json TEXT,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_docs_table_sql)
            cursor.execute(create_queries_table_sql)
            self.conn.commit()
            print("Database initialized successfully.")
        except sqlite3.Error as e:
            print(f"Error initializing database tables: {e}")

    def add_document_record(
        self,
        filename,
        vision_model,
        chart_dir,
        faiss_path,
        chunks_path,
        chart_descriptions,
    ):
        """Adds a new processed document session to the database."""
        sql = """ INSERT INTO documents(original_filename, vision_model_used, timestamp, chart_dir, faiss_index_path, chunks_path, chart_descriptions_json)
                  VALUES(?,?,?,?,?,?,?) """
        cursor = self.conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chart_descriptions_str = json.dumps(chart_descriptions)
        cursor.execute(
            sql,
            (
                filename,
                vision_model,
                ts,
                chart_dir,
                faiss_path,
                chunks_path,
                chart_descriptions_str,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_query_record(self, doc_id, question, response, sources):
        """Adds a new question-answer record linked to a document."""
        sql = """ INSERT INTO queries(document_id, question, response, sources_json, timestamp)
                  VALUES(?,?,?,?,?) """
        cursor = self.conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sources_str = json.dumps(sources)
        cursor.execute(sql, (doc_id, question, response, sources_str, ts))
        self.conn.commit()
        return cursor.lastrowid

    def get_all_documents(self):
        """Retrieves all past document sessions, ordered by most recent."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, original_filename, timestamp FROM documents ORDER BY timestamp DESC"
        )
        return cursor.fetchall()

    def get_document_by_id(self, doc_id):
        """Retrieves a single document's metadata by its ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None

        # Convert row tuple to a dictionary for easier access
        doc = {
            "id": row[0],
            "original_filename": row[1],
            "vision_model_used": row[2],
            "timestamp": row[3],
            "chart_dir": row[4],
            "faiss_index_path": row[5],
            "chunks_path": row[6],
            "chart_descriptions": json.loads(row[7]),
        }
        return doc

    def get_queries_for_document(self, doc_id):
        """Retrieves the full Q&A history for a given document ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT question, response, sources_json, timestamp FROM queries WHERE document_id = ? ORDER BY timestamp ASC",
            (doc_id,),
        )
        rows = cursor.fetchall()

        history = []
        for row in rows:
            history.append(
                {
                    "question": row[0],
                    "response": row[1],
                    "sources": json.loads(row[2]),
                    "timestamp": row[3],
                }
            )
        return history

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
