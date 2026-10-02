import os
import uuid
from pathlib import Path

from typing import Any, Dict, List

import numpy as np
from langchain_core.documents import Document
import logging
print("Finished importing basic modules")

from sentence_transformers import SentenceTransformer

print("Finished importing sentence transformer modules")
from sklearn.metrics.pairwise import cosine_similarity

print("Finished importing sklearn modules")
from synthetic_patient_data_retrival.DatabaseManager import PatientDatabaseManager
from synthetic_patient_data_retrival.rag_metadata import rows_to_embed_batch
print("Finished importing metadata modules")

### Embeddings and VectorStoreDB
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load a SentenceTransformer model for encoding text.

        Args:
            model_name: HuggingFace model name for sentence embeddings.

        Return:
            None.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Instantiate self.model from self.model_name.

        Args:
            None.

        Return:
            None.
        """
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    @property
    def embedding_dim(self) -> int:
        """
        Embedding width of the loaded model.

        Args:
            None.

        Return:
            Vector dimension (e.g. 384).
        """
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            return self.model.get_sentence_embedding_dimension()
        return self.model.get_embedding_dimension()

    def generate_embeddings(
        self, texts: List[str], as_float32_list: bool = True) -> List[List[float]] | np.ndarray:
        """
        Encode texts with the loaded SentenceTransformer model.

        Args:
            texts: Strings to embed.
            as_float32_list: If True, each vector is a float32 list for sqlite-vec;
                if False, a single float32 ndarray of shape (len(texts), embedding_dim).

        Return:
            List of embedding vectors, or a numpy array when as_float32_list is False.
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        if not texts:
            if as_float32_list:
                return []
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 32,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if as_float32_list:
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

### RAG Retriever Pipeline from VectorDB
class RAGRetriever:
    """Index clinical rows into vec0 and run FTS / KNN retrieval."""

    def __init__(self, dbManager: PatientDatabaseManager, embedding_manager: EmbeddingManager, log_file: str = 'app.log'):
        """
        Wire database access, embeddings, and file logging.

        Args:
            dbManager: PatientDatabaseManager (SQLite + stores).
            embedding_manager: Encoder for query and batch embeds.
            log_file: Path for debug log output.

        Return:
            None.
        """
        self.dbManager = dbManager
        self.embedding_manager = embedding_manager
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def collect_all_embed_batch(self) -> tuple[list[str], list[tuple[Any, ...]]]:
        """
        Build embed texts and rag_chunks tuples for every clinical table.

        Args:
            None.

        Return:
            Parallel lists of embed strings and (chunk_id, source_id, resource_type) rows.
        """
        texts: list[str] = []
        chunk_rows: list[tuple[Any, ...]] = []
        sources = [
            (self.dbManager.get_patient_data(), "patient", "patients"),
            (self.dbManager.get_encounter_data(), "encounter", "encounters"),
            (self.dbManager.get_condition_data(), "condition", "conditions"),
            (self.dbManager.get_observation_data(), "observation", "observations"),
            (self.dbManager.get_medication_data(), "medication", "medications"),
            (self.dbManager.get_procedure_data(), "procedure", "procedures"),
        ]
        for rows, resource_type, label in sources:
            batch_texts, batch_rows = rows_to_embed_batch(rows, resource_type, label)
            texts.extend(batch_texts)
            chunk_rows.extend(batch_rows)
        return texts, chunk_rows

    def add_documents(self, files: List[str]):
        """
        Load Synthea FHIR bundles into the relational store.

        Args:
            files: FHIR JSON file paths to ingest.

        Return:
            None.
        """
        self.dbManager.load_patient_data(files=files, patient_data=None, patientsCount=None)

    def rebuild_embeddings_fts_tables(self):
        """
        Recreate vector tables, embed all rows, and rebuild identity FTS.

        Args:
            reset: Must be True; drops rag_chunks / vec0 and re-indexes everything.

        Return:
            None.
        """

        self.dbManager.create_vector_tables(vector_size=self.embedding_manager.embedding_dim)
        self.dbManager.conn.commit()
        texts, chunk_rows = self.collect_all_embed_batch()
        if not texts:
            raise ValueError("No rows to embed")

        vectors = self.embedding_manager.generate_embeddings(texts)
        self.dbManager.add_embeddings(chunk_rows, vectors)
        self.dbManager.rebuild_patients_fts()
        self.dbManager.rebuild_encounters_fts()
        self.dbManager.conn.commit()


    def retrieve_documents_by_knn(self, query: str = None, query_embedding: list[float] = None, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve chunk hits by sqlite-vec KNN on the query embedding.

        Args:
            query: Natural-language query; encoded when query_embedding is omitted.
            query_embedding: Precomputed query vector (float32 list).
            top_k: Maximum neighbors to return.
            score_threshold: Minimum similarity score threshold

        Return:
            Hydrated records from get_document per chunk_id (empty list on error).
        """
        if not query and not query_embedding:
            raise ValueError("Either query or query_embedding must be provided")
            
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        documents = [] # List of Langchain Documents with content and metadata
        # Generate query embedding
        if not query_embedding:
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            chunk_ids = self.dbManager.retrieve_k_nearest_neighbors(query_embedding, top_k)
            self.logger.info(f"Chunk IDs: {chunk_ids}")
            for chunk_id in chunk_ids:
                document = self.dbManager.get_document(chunk_id)
                documents.append(document)
                self.logger.info(f"Document: {document}")
            return documents
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return []


    def retrieve_documents_by_fts(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Shortlist patients and encounters with BM25 identity FTS.

        Args:
            query: Natural-language query.
            top_k: Max hits per patients_fts and encounters_fts.
            score_threshold: Minimum similarity score threshold

        Return:
            Combined list of patient and encounter FTS hit dicts.
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Search in vector store
        try:
            results = self.dbManager.search_patients_fts(query, top_k)
            self.logger.info(f"Patients results: ")
            for result in results:
                self.logger.info(result)
            encounters_results = self.dbManager.search_encounters_fts(query, top_k)
            self.logger.info(f"Encounters results: ")
            for result in encounters_results:
                self.logger.info(result)
            results.extend(encounters_results)
            return results
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return []

    def retrieve_documents_hybrid(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query using a hybrid approach of KNN and FTS
        
        Args:
            query: Natural-language query.
            top_k: Passed through to KNN and FTS helpers.
            score_threshold: Minimum similarity score threshold

        Return:
            Combined list of KNN and FTS hit dicts.
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Retrieve documents using KNN
        self.logger.info(f"KNN documents: ")
        knn_documents = self.retrieve_documents_by_knn(query_embedding=query_embedding, top_k=top_k, score_threshold=score_threshold)
        # Retrieve documents using FTS
        self.logger.info(f"FTS documents: ")
        fts_documents = self.retrieve_documents_by_fts(query=query, top_k=top_k, score_threshold=score_threshold)




dbManager = PatientDatabaseManager()
embedding_manager = EmbeddingManager()
rag_retriever = RAGRetriever(dbManager, embedding_manager)
rag_retriever.rebuild_embeddings_fts_tables(reset=True)
resultsFTS5 = rag_retriever.retrieve_documents_by_fts('When did Zada last visit the hospital?', top_k=5, score_threshold=0.0)

# print(f"Total documents: {len(all_documents)}")

# for doc in all_documents:
#     if len(doc.page_content) <29:
#         print(doc.page_content)

"""
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(encounters)

all_pdf_documents = process_all_pdfs("../data")
chunks = split_documents(documents=all_pdf_documents)

### Intialize Embedding Manager

embedding_manager = EmbeddingManager()

vectorstore = VectorStore()

texts = [doc.page_content for doc in chunks]

### Generate the Embeddings
embeddings = embedding_manager.generate_embeddings(texts)

### Store Embeddings into a Chroma VectorStore
vectorstore.add_documents(chunks, embeddings)

rag_retriever = RAGRetriever(vectorstore, embedding_manager)

rag_retriever.retrieve('Work Experience as a Medpace Data Engineer')
"""