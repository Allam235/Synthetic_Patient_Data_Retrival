import hashlib

import sqlite_vec
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, conn):
        self.conn = conn
        self.vector_size: int | None = None

    @staticmethod
    def get_stable_chunk_id(resource_type: str, source_id: str=None, document: Document=None) -> int:
        """
        Hash (resource_type, source_id) to a positive int64 vec0 rowid.

        Args:
            resource_type: Clinical resource type string.
            source_id: Primary key of the resource row; preferred when set.
            document: Deprecated LangChain document fallback when source_id is omitted.

        Return:
            Stable chunk_id used as rag_chunks.chunk_id and vec0 rowid.
        """
        if resource_type is None:
            raise ValueError("Resource type is required")
        if source_id is None and document is None:
            raise ValueError("Source ID or document is required")
        if source_id is None:
            match resource_type:
                case "patient":
                    source_id = document.patient_id
                case "encounter":
                    source_id = document.encounter_id
                case "condition":
                    source_id = document.condition_id
                case "medication":
                    source_id = document.medication_id
                case "procedure":
                    source_id = document.procedure_id
                case _:
                    raise ValueError(f"Invalid resource type or resource_id not in Document: {resource_type}")
            
        key = f"{resource_type}\0{source_id}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF

    def create_vector_tables(self, vector_size: int) -> None:
        """
        Drop and recreate rag_chunks plus vec0 for the given embedding width.

        Args:
            vector_size: Embedding dimension (e.g. 384 for all-MiniLM-L6-v2).

        Return:
            None.
        """
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
                UNIQUE (resource_type, source_id)
            );

            CREATE INDEX idx_rag_chunks_resource_type ON rag_chunks(resource_type);
            CREATE INDEX idx_rag_chunks_source ON rag_chunks(resource_type, source_id);

            CREATE VIRTUAL TABLE rag_chunk_embeddings USING vec0(
                embedding float[{vector_size}]
            );
            """.format(vector_size=vector_size)
        )

    def add_embeddings(self, chunk_rows, vectors) -> None:
        """
        Upsert rag_chunks pointers and vec0 embeddings in one transaction.

        Args:
            chunk_rows: Rows of (chunk_id, source_id, resource_type).
            vectors: float32 embedding lists, same length as chunk_rows.

        Return:
            None.
        """
        if len(chunk_rows) != len(vectors):
            raise ValueError("chunk_rows and vectors must have the same length")

        cursor = self.conn.cursor()
        sql_statement_chunks = """
            INSERT INTO rag_chunks
                (chunk_id,
                source_id,
                resource_type
                )
            VALUES (?, ?, ?)
            ON CONFLICT (resource_type, source_id) DO UPDATE SET
                chunk_id = excluded.chunk_id,
                source_id = excluded.source_id,
                resource_type = excluded.resource_type
        """
        sql_statement_embeddings = """
            INSERT OR REPLACE INTO rag_chunk_embeddings
                (rowid, embedding)
            VALUES (?, ?)
        """
        vec_rows = []
        for chunk_row, vector in zip(chunk_rows, vectors):
            chunk_id = chunk_row[0]
            blob = sqlite_vec.serialize_float32(vector)
            vec_rows.append((chunk_id, blob))

        with self.conn:
            cursor.executemany(sql_statement_chunks, chunk_rows)
            cursor.executemany(sql_statement_embeddings, vec_rows)

    def retrieve_k_nearest_neighbors(self, query_vector: list[float], k: int) -> list[int]:
        """
        KNN search on rag_chunk_embeddings; return matching chunk_ids.

        Args:
            query_vector: Query embedding as a float32-valued list.
            k: Number of neighbors (vec0 MATCH k parameter).

        Return:
            chunk_id values (vec0 rowids) ordered by increasing distance.
        """
        cursor = self.conn.cursor()
        query_bytes = sqlite_vec.serialize_float32(query_vector)
        cursor.execute(
            """
            SELECT 
                rowid,
                distance as score,
                chunks.resource_type,
                chunks.*
            FROM rag_chunk_embeddings as vec
            LEFT JOIN rag_chunks as chunks ON vec.rowid = chunks.chunk_id
            WHERE embedding MATCH ? AND k = ?
            """, 
            (query_bytes, k)
        )
        
        # 3. Extract the rowids (row[0] because fetchall returns a list of tuples)
        return [row[0] for row in cursor.fetchall()]

