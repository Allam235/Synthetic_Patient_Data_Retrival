import os
import uuid
from pathlib import Path

from typing import Any, Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document
import logging
print("Finished importing basic modules")

from sentence_transformers import SentenceTransformer

print("Finished importing sentence transformer modules")
from sklearn.metrics.pairwise import cosine_similarity

print("Finished importing sklearn modules")
from synthetic_patient_data_retrival.DatabaseManager import PatientDatabaseManager
from synthetic_patient_data_retrival.rag_metadata import (
    document_to_rag_chunk_tuple,
    rows_to_documents,
)
print("Finished importing metadata modules")

### Helper Functions to convert SQLite rows to Langchain Documents



### Embeddings and VectorStoreDB
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize function for the embedding manager
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model using the declared HuggingFace model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    @property
    def embedding_dim(self) -> int:
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            return self.model.get_sentence_embedding_dimension()
        return self.model.get_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        if not texts:
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
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

### RAG Retriever Pipeline from VectorDB
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, dbManager: PatientDatabaseManager, embedding_manager: EmbeddingManager, log_file: str = 'app.log'):
        """
        Initialize the retriever
        
        Args:
            dbManager: Database manager containing document embeddings
            embedding_manager: Manager for generating query embeddings
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

    @staticmethod
    def patients_to_documents(patients):
        return rows_to_documents(patients, "patient", "patients")

    @staticmethod
    def encounters_to_documents(encounters):
        return rows_to_documents(encounters, "encounter", "encounters")

    @staticmethod
    def conditions_to_documents(conditions):
        return rows_to_documents(conditions, "condition", "conditions")

    @staticmethod
    def observations_to_documents(observations):
        return rows_to_documents(observations, "observation", "observations")

    @staticmethod
    def medications_to_documents(medications):
        return rows_to_documents(medications, "medication", "medications")

    @staticmethod
    def procedures_to_documents(procedures):
        return rows_to_documents(procedures, "procedure", "procedures")

    def process_all_documents(self) -> List[Document]:
        """
        Process all documents from the database into a list of langchain documents
        
        Returns:
            List of langchain documents
        """
        patient_documents = self.patients_to_documents(self.dbManager.get_patient_data())
        encounter_documents = self.encounters_to_documents(self.dbManager.get_encounter_data())
        condition_documents = self.conditions_to_documents(self.dbManager.get_condition_data())
        observation_documents = self.observations_to_documents(self.dbManager.get_observation_data())
        medication_documents = self.medications_to_documents(self.dbManager.get_medication_data())
        procedure_documents = self.procedures_to_documents(self.dbManager.get_procedure_data())
        return patient_documents + encounter_documents + condition_documents + observation_documents + medication_documents + procedure_documents


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            #Process the resulting queried embeddings
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def add_documents(self, files: List[str]):
        """
        Add FHIR JSON files to the database
        
        Args:
            files: List of file paths to add
        """
        self.dbManager.load_patient_data(files=files, patient_data=None, patientsCount=None)

    def add_embeddings(self, documents: List[Document] = None, reset: bool = False):
        """
        Add embeddings to the database(embeds incoming documents if provided)
        
        Args:
            documents: List of langchain documents to add
            reset: Whether to reembed all the existing embeddings(embeds incoming vectors if provided)
        """
        batch: List[Document] = []
        if documents:
            batch.extend(documents)
        if reset:
            self.dbManager.create_vector_tables(
                vector_size=self.embedding_manager.embedding_dim,
            )
            self.dbManager.conn.commit()
            batch.extend(self.process_all_documents())
        if not batch:
            raise ValueError("add_embeddings requires documents and/or reset=True")

        texts = [doc.page_content for doc in batch]
        vectors = self.embedding_manager.generate_embeddings(texts)
        chunk_rows = [
            document_to_rag_chunk_tuple(doc, self.embedding_manager.model_name)
            for doc in batch
        ]
        self.dbManager.add_embeddings(chunk_rows, vectors)
        self.dbManager.rebuild_patients_fts()
        self.dbManager.rebuild_encounters_fts()
        self.dbManager.conn.commit()

    def retrieve_metadata_filters(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query with metadata filters
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
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





dbManager = PatientDatabaseManager()
embedding_manager = EmbeddingManager()
rag_retriever = RAGRetriever(dbManager, embedding_manager)
rag_retriever.add_embeddings(reset=True)
resultsFTS5 = rag_retriever.retrieve_metadata_filters('When did Zada last visit the hospital?', top_k=5, score_threshold=0.0)


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