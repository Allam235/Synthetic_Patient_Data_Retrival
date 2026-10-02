import re
from pathlib import Path
from typing import Any


class FTSStore:
    def __init__(self, conn):
        self.conn = conn

    @staticmethod
    def create_fts_tables(cursor) -> None:
        FTSStore.create_patients_fts(cursor)
        FTSStore.create_encounters_fts(cursor)

    @staticmethod
    def create_patients_fts(cursor) -> None:
        cursor.executescript(
            """
            DROP TABLE IF EXISTS patients_fts;
            CREATE VIRTUAL TABLE patients_fts USING fts5(
                identity_text,
                patient_id UNINDEXED,
                first_name UNINDEXED,
                last_name UNINDEXED,
                birth_date UNINDEXED,
                gender UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )

    @staticmethod
    def create_encounters_fts(cursor) -> None:
        cursor.executescript(
            """
            DROP TABLE IF EXISTS encounters_fts;
            CREATE VIRTUAL TABLE encounters_fts USING fts5(
                identity_text,
                encounter_id UNINDEXED,
                patient_id UNINDEXED,
                encounter_date UNINDEXED,
                encounter_type UNINDEXED,
                reason UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )

    @staticmethod
    def patient_identity_text(row) -> str:
        parts = [
            row["patient_id"],
            row["first_name"],
            row["last_name"],
            row["birth_date"],
            row["gender"],
            re.sub(r"\d+", "", row["first_name"] or ""),
            re.sub(r"\d+", "", row["last_name"] or ""),
        ]
        if row["source_file"]:
            parts.append(Path(row["source_file"]).stem.replace("_", " "))
        return " ".join(p for p in parts if p)

    @staticmethod
    def encounter_identity_text(row) -> str:
        parts = [
            row["encounter_id"],
            row["patient_id"],
            row["encounter_date"],
            row["encounter_type"],
            row["reason"],
            row["first_name"],
            row["last_name"],
            re.sub(r"\d+", "", row["first_name"] or ""),
            re.sub(r"\d+", "", row["last_name"] or ""),
        ]
        if row["source_file"]:
            parts.append(Path(row["source_file"]).stem.replace("_", " "))
        return " ".join(p for p in parts if p)

    @staticmethod
    def tokenize_query_for_fts(query: str) -> str | None:
        tokens = re.findall(r"[A-Za-z0-9\-]+", query)
        tokens = [t for t in tokens if len(t) >= 2]
        if not tokens:
            return None
        return " OR ".join(f'"{t}"' for t in tokens)

    def rebuild_patients_fts(self) -> None:
        cursor = self.conn.cursor()
        self.create_patients_fts(cursor)
        cursor.execute(
            "SELECT patient_id, first_name, last_name, birth_date, gender, source_file FROM patients"
        )
        rows = [
            (
                self.patient_identity_text(row),
                row["patient_id"],
                row["first_name"],
                row["last_name"],
                row["birth_date"],
                row["gender"],
            )
            for row in cursor.fetchall()
        ]
        if rows:
            cursor.executemany(
                """
                INSERT INTO patients_fts (
                    identity_text, patient_id, first_name, last_name, birth_date, gender
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def rebuild_encounters_fts(self) -> None:
        cursor = self.conn.cursor()
        self.create_encounters_fts(cursor)
        cursor.execute(
            """
            SELECT
                e.encounter_id,
                e.patient_id,
                e.encounter_date,
                e.encounter_type,
                e.reason,
                e.source_file,
                p.first_name,
                p.last_name
            FROM encounters AS e
            LEFT JOIN patients AS p ON e.patient_id = p.patient_id
            """
        )
        rows = [
            (
                self.encounter_identity_text(row),
                row["encounter_id"],
                row["patient_id"],
                row["encounter_date"],
                row["encounter_type"],
                row["reason"],
            )
            for row in cursor.fetchall()
        ]
        if rows:
            cursor.executemany(
                """
                INSERT INTO encounters_fts (
                    identity_text,
                    encounter_id,
                    patient_id,
                    encounter_date,
                    encounter_type,
                    reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def search_patients_fts(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        match_expr = self.tokenize_query_for_fts(query)
        if not match_expr:
            return []
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT patient_id, first_name, last_name, birth_date, gender, bm25(patients_fts) AS rank
            FROM patients_fts
            WHERE patients_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_expr, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def search_encounters_fts(
        self,
        query: str,
        limit: int = 3,
        patient_id: str | None = None,
    ) -> list[dict[str, Any]]:
        match_expr = self.tokenize_query_for_fts(query)
        if not match_expr:
            return []
        cursor = self.conn.cursor()
        if patient_id is None:
            cursor.execute(
                """
                SELECT
                    encounter_id,
                    patient_id,
                    encounter_date,
                    encounter_type,
                    reason,
                    bm25(encounters_fts) AS rank
                FROM encounters_fts
                WHERE encounters_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, limit),
            )
        else:
            cursor.execute(
                """
                SELECT
                    encounter_id,
                    patient_id,
                    encounter_date,
                    encounter_type,
                    reason,
                    bm25(encounters_fts) AS rank
                FROM encounters_fts
                WHERE encounters_fts MATCH ?
                  AND patient_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, patient_id, limit),
            )
        return [dict(row) for row in cursor.fetchall()]
