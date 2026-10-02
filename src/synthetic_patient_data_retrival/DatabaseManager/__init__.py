from pathlib import Path

from synthetic_patient_data_retrival.DatabaseManager.connection import DBConnection
from synthetic_patient_data_retrival.DatabaseManager.fts_store import FTSStore
from synthetic_patient_data_retrival.DatabaseManager.original_store import OriginalStore
from synthetic_patient_data_retrival.DatabaseManager.vector_store import VectorStore

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _PROJECT_ROOT / "data" / "sqlite" / "patient.db"
_DEFAULT_FHIR = _PROJECT_ROOT / "data" / "generator" / "output" / "fhir"


class PatientDatabaseManager:
    """Facade: one connection, relational + FTS + vector stores."""

    def __init__(
        self,
        db_path=_DEFAULT_DB,
        gen_output_path=_DEFAULT_FHIR,
    ):
        self.db_path = Path(db_path)
        self.gen_output_path = Path(gen_output_path)
        self._connection = DBConnection(self.db_path)
        self.conn = self._connection.get_connection()
        self.cursor = self._connection.cursor

        self.original = OriginalStore(self.conn, self.gen_output_path)
        self.fts = FTSStore(self.conn)
        self.vector = VectorStore(self.conn)

    @property
    def vector_size(self):
        return self.vector.vector_size

    @vector_size.setter
    def vector_size(self, value):
        self.vector.vector_size = value

    def close(self):
        self._connection.close()
        self.conn = None
        self.cursor = None

    def intialize_database(self):
        self.original.create_tables(self.cursor)
        self.fts.create_fts_tables(self.cursor)
        self.conn.commit()

    def refresh_patient_data(self):
        self.intialize_database()

    @staticmethod
    def get_stable_chunk_id(resource_type: str, source_id: str) -> int:
        return VectorStore.get_stable_chunk_id(resource_type, source_id)

    @staticmethod
    def tokenize_query_for_fts(query: str) -> str | None:
        return FTSStore.tokenize_query_for_fts(query)

    def load_patient_data(self, patient_data=None, patientsCount=None, files=None):
        self.original.load_patient_data(
            patient_data=patient_data,
            patientsCount=patientsCount,
            files=files,
        )
        self.fts.rebuild_patients_fts()
        self.fts.rebuild_encounters_fts()
        self.conn.commit()

    def create_vector_tables(self, vector_size: int):
        self.vector.create_vector_tables(vector_size)

    def add_embeddings(self, chunk_rows, vectors):
        self.vector.add_embeddings(chunk_rows, vectors)

    def check_loaded_data(self):
        return self.original.check_loaded_data()

    def process_generator_output(self, patientsCount=None, files=None):
        return self.original.process_generator_output(
            patientsCount=patientsCount, files=files
        )

    def get_patient_data(self, patient_id=None):
        return self.original.get_patient_data(patient_id)

    def get_encounter_data(self, encounter_id=None):
        return self.original.get_encounter_data(encounter_id)

    def get_condition_data(self, condition_id=None):
        return self.original.get_condition_data(condition_id)

    def get_observation_data(self, observation_id=None):
        return self.original.get_observation_data(observation_id)

    def get_medication_data(self, medication_id=None):
        return self.original.get_medication_data(medication_id)

    def get_procedure_data(self, procedure_id=None):
        return self.original.get_procedure_data(procedure_id)

    def rebuild_patients_fts(self):
        self.fts.rebuild_patients_fts()

    def rebuild_encounters_fts(self):
        self.fts.rebuild_encounters_fts()

    def search_patients_fts(self, query: str, limit: int = 3):
        return self.fts.search_patients_fts(query, limit=limit)

    def search_encounters_fts(self, query: str, limit: int = 3, patient_id=None):
        return self.fts.search_encounters_fts(
            query, limit=limit, patient_id=patient_id
        )

    def retrieve_k_nearest_neighbors(self, query_vector: list[float], k: int):
        return self.vector.retrieve_k_nearest_neighbors(query_vector, k)


StorageManager = PatientDatabaseManager

__all__ = [
    "DBConnection",
    "FTSStore",
    "OriginalStore",
    "PatientDatabaseManager",
    "StorageManager",
    "VectorStore",
]
