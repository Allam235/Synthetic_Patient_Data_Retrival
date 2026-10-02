import hashlib

import numpy as np
import sqlite_vec


class VectorStore:
    def __init__(self, conn):
        self.conn = conn
        self.vector_size: int | None = None

    @staticmethod
    def get_stable_chunk_id(resource_type: str, source_id: str) -> int:
        key = f"{resource_type}\0{source_id}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF

    def create_vector_tables(self, vector_size: int) -> None:
        self.vector_size = vector_size
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            DROP TABLE IF EXISTS rag_chunk_embeddings;
            DROP TABLE IF EXISTS rag_chunks;

            CREATE TABLE rag_chunks (
                chunk_id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                encounter_id TEXT,
                code TEXT,
                event_date TEXT,
                source_file TEXT,
                content TEXT NOT NULL,
                embedding_model TEXT,
                metadata_json TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                UNIQUE (resource_type, source_id)
            );

            CREATE INDEX idx_rag_chunks_patient_id ON rag_chunks(patient_id);
            CREATE INDEX idx_rag_chunks_encounter_id ON rag_chunks(encounter_id);
            CREATE INDEX idx_rag_chunks_resource_type ON rag_chunks(resource_type);
            CREATE INDEX idx_rag_chunks_source ON rag_chunks(resource_type, source_id);

            CREATE VIRTUAL TABLE rag_chunk_embeddings USING vec0(
                embedding float[{vector_size}]
            );
            """.format(vector_size=vector_size)
        )

    def add_embeddings(self, chunk_rows, vectors) -> None:
        if len(chunk_rows) != len(vectors):
            raise ValueError("chunk_rows and vectors must have the same length")

        cursor = self.conn.cursor()
        sql_statement_chunks = """
            INSERT INTO rag_chunks
                (chunk_id,
                source_id,
                resource_type,
                patient_id,
                encounter_id,
                code,
                event_date,
                source_file,
                content,
                embedding_model,
                metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (resource_type, source_id) DO UPDATE SET
                chunk_id = excluded.chunk_id,
                patient_id = excluded.patient_id,
                encounter_id = excluded.encounter_id,
                code = excluded.code,
                event_date = excluded.event_date,
                source_file = excluded.source_file,
                content = excluded.content,
                embedding_model = excluded.embedding_model,
                metadata_json = excluded.metadata_json
        """
        sql_statement_embeddings = """
            INSERT OR REPLACE INTO rag_chunk_embeddings
                (rowid, embedding)
            VALUES (?, ?)
        """
        vec_rows = []
        for chunk_row, vector in zip(chunk_rows, vectors):
            chunk_id = chunk_row[0]
            flat = np.asarray(vector, dtype=np.float32).reshape(-1)
            blob = sqlite_vec.serialize_float32(flat.tolist())
            vec_rows.append((chunk_id, blob))

        with self.conn:
            cursor.executemany(sql_statement_chunks, chunk_rows)
            cursor.executemany(sql_statement_embeddings, vec_rows)
