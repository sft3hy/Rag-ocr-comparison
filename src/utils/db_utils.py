import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


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
        self._migrate_database()  # Check and apply migrations

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

        # Sessions table
        create_sessions_table_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        );
        """

        # Documents table
        create_docs_table_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            original_filename TEXT NOT NULL,
            vision_model_used TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            chart_dir TEXT,
            faiss_index_path TEXT,
            chunks_path TEXT,
            chart_descriptions_json TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        );
        """

        # Queries table (New Schema)
        create_queries_table_sql = """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            sources_json TEXT,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        );
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute(create_sessions_table_sql)
            cursor.execute(create_docs_table_sql)
            cursor.execute(create_queries_table_sql)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"✗ Error initializing database tables: {e}")

    def _migrate_database(self):
        """
        Checks for legacy schemas (old columns/constraints) and migrates them.
        Specifically fixes 'queries.document_id' NOT NULL constraint by recreating the table.
        """
        if not self.conn:
            return

        cursor = self.conn.cursor()

        try:
            # 1. Check 'queries' table for legacy 'document_id' column
            cursor.execute("PRAGMA table_info(queries)")
            columns = [info[1] for info in cursor.fetchall()]

            # If we find 'document_id', it is the old schema.
            # We must recreate the table to remove the NOT NULL constraint.
            if "document_id" in columns:
                print(
                    "⚡ Migrating DB: Detected legacy 'queries' table. Archiving and recreating..."
                )

                # Rename old table to backup
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                rename_sql = f"ALTER TABLE queries RENAME TO queries_legacy_{timestamp}"
                cursor.execute(rename_sql)

                # Create the new table immediately
                create_queries_table_sql = """
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    response TEXT NOT NULL,
                    sources_json TEXT,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                """
                cursor.execute(create_queries_table_sql)
                print(
                    f"✓ Created new 'queries' table. Old data archived in 'queries_legacy_{timestamp}'."
                )

            # 2. Check 'documents' table for 'session_id' column
            cursor.execute("PRAGMA table_info(documents)")
            doc_columns = [info[1] for info in cursor.fetchall()]
            if "session_id" not in doc_columns:
                print("⚡ Migrating DB: Adding 'session_id' to 'documents' table...")
                cursor.execute(
                    "ALTER TABLE documents ADD COLUMN session_id INTEGER REFERENCES sessions(id)"
                )

            self.conn.commit()

        except sqlite3.Error as e:
            print(f"✗ Database migration warning: {e}")

    def create_session(self, filenames: List[str]) -> int:
        """Creates a new session for one or more documents."""
        if len(filenames) == 1:
            session_name = filenames[0]
        else:
            session_name = f"{filenames[0]} + {len(filenames) - 1} more"

        sql = "INSERT INTO sessions(session_name, timestamp) VALUES(?,?)"
        cursor = self.conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(sql, (session_name, ts))
        self.conn.commit()
        return cursor.lastrowid

    def add_document_record(
        self,
        filename: str,
        vision_model: str,
        chart_dir: str,
        faiss_path: str,
        chunks_path: str,
        chart_descriptions: Dict,
        session_id: Optional[int] = None,
    ) -> int:
        """Adds a new processed document to the database."""
        sql = """INSERT INTO documents(session_id, original_filename, vision_model_used, 
                 timestamp, chart_dir, faiss_index_path, chunks_path, chart_descriptions_json)
                 VALUES(?,?,?,?,?,?,?,?)"""
        cursor = self.conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chart_descriptions_str = json.dumps(chart_descriptions)
        cursor.execute(
            sql,
            (
                session_id,
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

    def add_query_record(
        self, session_id: int, question: str, response: str, sources: List[Dict]
    ) -> int:
        """Adds a new question-answer record linked to a session."""
        sql = """INSERT INTO queries(session_id, question, response, sources_json, timestamp)
                 VALUES(?,?,?,?,?)"""
        cursor = self.conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sources_str = json.dumps(sources)
        cursor.execute(sql, (session_id, question, response, sources_str, ts))
        self.conn.commit()
        return cursor.lastrowid

    def get_all_sessions(self) -> List[Tuple[int, str, str, int]]:
        """Retrieves all past sessions with document count."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.session_name, s.timestamp, COUNT(d.id) as doc_count
            FROM sessions s
            LEFT JOIN documents d ON s.id = d.session_id
            GROUP BY s.id
            ORDER BY s.timestamp DESC
        """
        )
        return cursor.fetchall()

    def get_session_documents(self, session_id: int) -> List[Dict]:
        """Retrieves all documents in a session."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE session_id = ?", (session_id,))
        rows = cursor.fetchall()

        documents = []
        for row in rows:
            documents.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "original_filename": row[2],
                    "vision_model_used": row[3],
                    "timestamp": row[4],
                    "chart_dir": row[5],
                    "faiss_index_path": row[6],
                    "chunks_path": row[7],
                    "chart_descriptions": json.loads(row[8]) if row[8] else {},
                }
            )
        return documents

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Retrieves a single document's metadata by its ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "session_id": row[1],
            "original_filename": row[2],
            "vision_model_used": row[3],
            "timestamp": row[4],
            "chart_dir": row[5],
            "faiss_index_path": row[6],
            "chunks_path": row[7],
            "chart_descriptions": json.loads(row[8]) if row[8] else {},
        }

    def get_queries_for_session(self, session_id: int) -> List[Dict]:
        """Retrieves the full Q&A history for a given session."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT question, response, sources_json, timestamp 
               FROM queries WHERE session_id = ? ORDER BY timestamp ASC""",
            (session_id,),
        )
        rows = cursor.fetchall()

        history = []
        for row in rows:
            history.append(
                {
                    "question": row[0],
                    "response": row[1],
                    "sources": json.loads(row[2]) if row[2] else [],
                    "timestamp": row[3],
                }
            )
        return history

    def get_session_info(self, session_id: int) -> Optional[Dict]:
        """Gets comprehensive information about a session."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return None

        documents = self.get_session_documents(session_id)
        queries = self.get_queries_for_session(session_id)

        return {
            "id": row[0],
            "session_name": row[1],
            "timestamp": row[2],
            "documents": documents,
            "query_count": len(queries),
        }

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
