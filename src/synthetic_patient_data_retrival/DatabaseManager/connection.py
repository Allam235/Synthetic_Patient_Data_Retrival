import sqlite3
from pathlib import Path

import sqlite_vec


class DBConnection:
    """Single SQLite connection: FK pragma, Row factory, sqlite-vec extension."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None

    def get_connection(self) -> sqlite3.Connection:
        if self.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
        return self.conn

    @property
    def cursor(self):
        return self.get_connection().cursor()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
