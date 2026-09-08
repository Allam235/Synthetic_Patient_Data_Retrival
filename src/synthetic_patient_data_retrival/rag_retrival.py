import os
import uuid
from pathlib import Path

from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

from synthetic_patient_data_retrival.loadGeneratorData import PatientDatabaseManager
### Read all PDFS in library




def encounters_to_documents(encounters):
    all_encounters = []
    print(f"Processing {len(encounters)} encounters")
    for encounter in encounters:
        try:
            doc = Document(
                page_content=f"{encounter['encounter_type']} | {encounter['reason']}",
                metadata={
                    'doc_id': encounter['encounter_id'],
                    'doc_type': 'encounter',
                    'patient_id': encounter['patient_id'],
                    'encounter_id': encounter['encounter_id'],
                    'encounter_date': encounter['encounter_date'],
                    'encounter_type': encounter['encounter_type'],
                    'reason': encounter['reason'],
                    'source_file': encounter['source_file'],
                }
            )
            all_encounters.append(doc)
        except Exception as e:
            print(f"Error processing encounter: {e}")
    print(f"Processed {len(all_encounters)} encounters")
    return all_encounters


def patients_to_documents(patients):
    all_patients = []
    print(f"Processing {len(patients)} patients")
    for patient in patients:
        try:
            doc = Document(
                page_content=f"{patient['first_name']} | {patient['last_name']} | {patient['gender']} | {patient['birth_date']}",
                metadata={
                    'doc_id': patient['patient_id'],
                    'doc_type': 'patient',
                    'patient_id': patient['patient_id'],
                    'first_name': patient['first_name'],
                    'last_name': patient['last_name'],
                    'birth_date': patient['birth_date'],
                    'gender': patient['gender'],
                    'source_file': patient['source_file'],
                }
            )
            all_patients.append(doc)
        except Exception as e:
            print(f"Error processing patient: {e}")
    print(f"Processed {len(all_patients)} patients")
    return all_patients


def conditions_to_documents(conditions):
    all_conditions = []
    print(f"Processing {len(conditions)} conditions")
    for condition in conditions:
        try:
            doc = Document(
                page_content=f"{condition['description']}",
                metadata={
                    'doc_id': condition['condition_id'],
                    'doc_type': 'condition',
                    'patient_id': condition['patient_id'],
                    'encounter_id': condition['encounter_id'],
                    'condition_id': condition['condition_id'],
                    'code': condition['code'],
                    'description': condition['description'],
                    'onset_date': condition['onset_date'],
                    'source_file': condition['source_file']
                }
            )
            all_conditions.append(doc)
        except Exception as e:
            print(f"Error processing condition: {e}")
    print(f"Processed {len(all_conditions)} conditions")
    return all_conditions


def observations_to_documents(observations):
    all_observations = []
    print(f"Processing {len(observations)} observations")
    for observation in observations:
        try:
            doc = Document(
                page_content=f"{observation['description']}: {observation['value']} {observation['unit']}",
                metadata={
                    'doc_id': observation['observation_id'],
                    'doc_type': 'observation',
                    'patient_id': observation['patient_id'],
                    'encounter_id': observation['encounter_id'],
                    'observation_id': observation['observation_id'],
                    'observation_date': observation['observation_date'],
                    'code': observation['code'],
                    'description': observation['description'],
                    'value': observation['value'],
                    'unit': observation['unit'],
                    'source_file': observation['source_file']
                }
            )
            all_observations.append(doc)
        except Exception as e:
            print(f"Error processing observation: {e}")
    print(f"Processed {len(all_observations)} observations")
    return all_observations


def medications_to_documents(medications):
    all_medications = []
    print(f"Processing {len(medications)} medications")
    for medication in medications:
        try:
            doc = Document(
                page_content=f"{medication['description']} | {medication['start_date']} | {medication['end_date']}",
                metadata={
                    'doc_id': medication['medication_id'],
                    'doc_type': 'medication',
                    'patient_id': medication['patient_id'],
                    'encounter_id': medication['encounter_id'],
                    'medication_id': medication['medication_id'],
                    'description': medication['description'],
                    'start_date': medication['start_date'],
                    'end_date': medication['end_date'],
                    'source_file': medication['source_file']
                }
            )
            all_medications.append(doc)
        except Exception as e:
            print(f"Error processing medication: {e}")
    print(f"Processed {len(all_medications)} medications")
    return all_medications


def procedures_to_documents(procedures):
    all_procedures = []
    print(f"Processing {len(procedures)} procedures")
    for procedure in procedures:
        try:
            doc = Document(
                page_content=f"{procedure['description']}",
                metadata={
                    'doc_id': procedure['procedure_id'],
                    'doc_type': 'procedure',
                    'patient_id': procedure['patient_id'],
                    'encounter_id': procedure['encounter_id'],
                    'procedure_id': procedure['procedure_id'],
                    'description': procedure['description'],
                    'procedure_date': procedure['procedure_date'],
                    'source_file': procedure['source_file']
                }
            )
            all_procedures.append(doc)
        except Exception as e:
            print(f"Error processing procedure: {e}")
    print(f"Processed {len(all_procedures)} procedures")
    return all_procedures


### Text splitting into chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    '''
    Splits a list of Langchain Documents into smaller chunks

    Args:
    documents: List of Langchain Documents
    chunk_size: Maximum size of each chunk (default: 1000 characters)
    chunk_overlap: Number of characters to overlap between chunks (default: 200 characters)
    '''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs


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
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

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
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

### VectorStore

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection(where the vector embeddings will be stored)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of chunked Langchained Documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


### RAG Retriever Pipeline from VectorDB

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

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

@staticmethod
def process_all_documents(dbManager):
    patient_documents = patients_to_documents(dbManager.get_patient_data())
    encounter_documents = encounters_to_documents(dbManager.get_encounter_data())
    condition_documents = conditions_to_documents(dbManager.get_condition_data())
    observation_documents = observations_to_documents(dbManager.get_observation_data())
    medication_documents = medications_to_documents(dbManager.get_medication_data())
    procedure_documents = procedures_to_documents(dbManager.get_procedure_data())
    return patient_documents + encounter_documents + condition_documents + observation_documents + medication_documents + procedure_documents


dbManager = PatientDatabaseManager()


all_documents = process_all_documents(dbManager)


print(f"Total documents: {len(all_documents)}")

for doc in all_documents:
    if len(doc.page_content) <29:
        print(doc.page_content)

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