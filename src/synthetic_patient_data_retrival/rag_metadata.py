"""
Embed text and rag_chunks pointer columns for the write (indexing) path.
Hydration after search uses resource_type + source_id via OriginalStore.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from synthetic_patient_data_retrival.DatabaseManager import PatientDatabaseManager

# Per-resource: SQL row id column + text sent to the embedding model.
RAG_RESOURCE_METADATA: Dict[str, Dict[str, Any]] = {
    "patient": {
        "doc_id_field": "patient_id",
        "page_content": lambda row: (
            f"{row['first_name']} | {row['last_name']} | {row['gender']} | {row['birth_date']}"
        ),
    },
    "encounter": {
        "doc_id_field": "encounter_id",
        "page_content": lambda row: (
            f"{row['encounter_type']} | {row['reason']}"
        ),
    },
    "condition": {
        "doc_id_field": "condition_id",
        "page_content": lambda row: f"{row['description']}",
    },
    "observation": {
        "doc_id_field": "observation_id",
        "page_content": lambda row: (
            f"{row['description']}: {row['value']} {row['unit']}"
        ),
    },
    "medication": {
        "doc_id_field": "medication_id",
        "page_content": lambda row: (
            f"{row['description']} | {row['start_date']} | {row['end_date']}"
        ),
    },
    "procedure": {
        "doc_id_field": "procedure_id",
        "page_content": lambda row: f"{row['description']}",
    },
}


def row_embed_text(row: Mapping[str, Any], resource_type: str) -> str:
    """
    Build the text string sent to the embedding model for one clinical row.

    Args:
        row: SQLite row mapping for a patient, encounter, or clinical fact.
        resource_type: Resource key in RAG_RESOURCE_METADATA (e.g. "condition").

    Return:
        Embed-ready page text for the row.
    """
    return RAG_RESOURCE_METADATA[resource_type]["page_content"](row)


def row_to_rag_chunk_tuple(row: Mapping[str, Any], resource_type: str) -> Tuple[Any, ...]:
    """
    Map a clinical row to a rag_chunks insert tuple (pointer only).

    Args:
        row: SQLite row mapping for the resource.
        resource_type: Resource key in RAG_RESOURCE_METADATA.

    Return:
        Tuple (chunk_id, source_id, resource_type) for vec0 / rag_chunks.
    """
    config = RAG_RESOURCE_METADATA[resource_type]
    source_id = row[config["doc_id_field"]]
    return (
        PatientDatabaseManager.get_stable_chunk_id(resource_type, source_id),
        source_id,
        resource_type,
    )


def rows_to_embed_batch(
    rows: Sequence[Mapping[str, Any]],
    resource_type: str,
    label: str,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """
    Convert many rows into parallel embed texts and rag_chunks tuples.

    Args:
        rows: Clinical table rows to index.
        resource_type: Resource key in RAG_RESOURCE_METADATA.
        label: Log label for the table (e.g. "observations").

    Return:
        Pair of lists: embed texts and (chunk_id, source_id, resource_type) rows.
    """
    texts: list[str] = []
    chunk_rows: list[tuple[Any, ...]] = []
    print(f"Processing {len(rows)} {label}")
    for row in rows:
        try:
            texts.append(row_embed_text(row, resource_type))
            chunk_rows.append(row_to_rag_chunk_tuple(row, resource_type))
        except Exception as e:
            print(f"Error processing {label}: {e}")
    print(f"Processed {len(chunk_rows)} {label}")
    return texts, chunk_rows


# ---------------------------------------------------------------------------
# DEPRECATED — LangChain Document + denormalized rag_chunks metadata snapshot.
# Retrieval should hydrate from OriginalStore using resource_type + source_id.
# ---------------------------------------------------------------------------

import json

from langchain_core.documents import Document

RAG_COMMON_METADATA: Dict[str, Dict[str, str]] = {
    "doc_id": {
        "description": "Primary key for the indexed resource; maps to rag_chunks.source_id.",
        "listed_in_documents": (
            "patient, encounter, condition, observation, medication, procedure"
        ),
        "rag_chunk_column": "source_id",
    },
    "doc_type": {
        "description": "Resource type string; maps to rag_chunks.resource_type.",
        "listed_in_documents": (
            "patient, encounter, condition, observation, medication, procedure"
        ),
        "rag_chunk_column": "resource_type",
    },
    "patient_id": {
        "description": "Owning patient FK.",
        "listed_in_documents": (
            "patient, encounter, condition, observation, medication, procedure"
        ),
        "rag_chunk_column": "patient_id",
    },
    "encounter_id": {
        "description": "Encounter FK when the fact is tied to a visit.",
        "listed_in_documents": (
            "encounter, condition, observation, medication, procedure"
        ),
        "rag_chunk_column": "encounter_id",
    },
    "code": {
        "description": "Clinical code (e.g. condition/observation code).",
        "listed_in_documents": "condition, observation",
        "rag_chunk_column": "code",
    },
    "source_file": {
        "description": "Synthea FHIR bundle filename.",
        "listed_in_documents": (
            "patient, encounter, condition, observation, medication, procedure"
        ),
        "rag_chunk_column": "source_file",
    },
}

RAG_CHUNK_COLUMN_FROM_METADATA: Dict[str, str] = {
    "doc_id": "source_id",
    "doc_type": "resource_type",
    "patient_id": "patient_id",
    "encounter_id": "encounter_id",
    "code": "code",
    "source_file": "source_file",
}

_DEPRECATED_RESOURCE_EXTRA: Dict[str, Dict[str, Any]] = {
    "patient": {
        "event_date_field": "birth_date",
        "metadata_fields": {
            "doc_id": "patient_id",
            "patient_id": "patient_id",
            "first_name": "first_name",
            "last_name": "last_name",
            "birth_date": "birth_date",
            "gender": "gender",
            "source_file": "source_file",
        },
        "metadata_json_fields": ("first_name", "last_name", "birth_date", "gender"),
    },
    "encounter": {
        "event_date_field": "encounter_date",
        "metadata_fields": {
            "doc_id": "encounter_id",
            "patient_id": "patient_id",
            "encounter_id": "encounter_id",
            "encounter_date": "encounter_date",
            "encounter_type": "encounter_type",
            "reason": "reason",
            "source_file": "source_file",
        },
        "metadata_json_fields": ("encounter_type", "reason"),
    },
    "condition": {
        "event_date_field": "onset_date",
        "metadata_fields": {
            "doc_id": "condition_id",
            "patient_id": "patient_id",
            "encounter_id": "encounter_id",
            "condition_id": "condition_id",
            "code": "code",
            "description": "description",
            "onset_date": "onset_date",
            "source_file": "source_file",
        },
        "metadata_json_fields": ("condition_id", "description", "onset_date"),
    },
    "observation": {
        "event_date_field": "observation_date",
        "metadata_fields": {
            "doc_id": "observation_id",
            "patient_id": "patient_id",
            "encounter_id": "encounter_id",
            "observation_id": "observation_id",
            "observation_date": "observation_date",
            "code": "code",
            "description": "description",
            "value": "value",
            "unit": "unit",
            "source_file": "source_file",
        },
        "metadata_json_fields": (
            "observation_id",
            "description",
            "value",
            "unit",
            "observation_date",
        ),
    },
    "medication": {
        "event_date_field": "start_date",
        "metadata_fields": {
            "doc_id": "medication_id",
            "patient_id": "patient_id",
            "encounter_id": "encounter_id",
            "medication_id": "medication_id",
            "description": "description",
            "start_date": "start_date",
            "end_date": "end_date",
            "source_file": "source_file",
        },
        "metadata_json_fields": ("medication_id", "description", "start_date", "end_date"),
    },
    "procedure": {
        "event_date_field": "procedure_date",
        "metadata_fields": {
            "doc_id": "procedure_id",
            "patient_id": "patient_id",
            "encounter_id": "encounter_id",
            "procedure_id": "procedure_id",
            "description": "description",
            "procedure_date": "procedure_date",
            "source_file": "source_file",
        },
        "metadata_json_fields": ("procedure_id", "description", "procedure_date"),
    },
}


def build_document_metadata(row: Mapping[str, Any], resource_type: str) -> Dict[str, Any]:
    """
    DEPRECATED: Build LangChain Document.metadata from a clinical row.

    Args:
        row: SQLite row mapping.
        resource_type: Resource key in _DEPRECATED_RESOURCE_EXTRA.

    Return:
        Metadata dict including doc_type and per-field copies from the row.
    """
    extra = _DEPRECATED_RESOURCE_EXTRA[resource_type]
    metadata: Dict[str, Any] = {"doc_type": resource_type}
    for meta_key, row_key in extra["metadata_fields"].items():
        metadata[meta_key] = row[row_key]
    return metadata


def row_to_document(row: Mapping[str, Any], resource_type: str) -> Document:
    """
    DEPRECATED: Wrap one clinical row as a LangChain Document.

    Args:
        row: SQLite row mapping.
        resource_type: Resource key for embed text and metadata.

    Return:
        Document with page_content and metadata snapshot.
    """
    return Document(
        page_content=row_embed_text(row, resource_type),
        metadata=build_document_metadata(row, resource_type),
    )


def metadata_json_from_document(doc: Document) -> str:
    """
    DEPRECATED: Serialize selected metadata fields to JSON for rag_chunks.

    Args:
        doc: LangChain Document with doc_type in metadata.

    Return:
        JSON string of metadata_json_fields for the resource type.
    """
    resource_type = doc.metadata["doc_type"]
    fields: Sequence[str] = _DEPRECATED_RESOURCE_EXTRA[resource_type]["metadata_json_fields"]
    payload = {
        key: doc.metadata[key]
        for key in fields
        if doc.metadata.get(key) is not None
    }
    return json.dumps(payload)


def document_to_rag_chunk_tuple(doc: Document, embedding_model: str) -> Tuple[Any, ...]:
    """
    DEPRECATED: Full denormalized rag_chunks tuple from a LangChain Document.

    Args:
        doc: Indexed document with doc_id, doc_type, and FK metadata.
        embedding_model: Model name stored on the chunk row.

    Return:
        Wide insert tuple including content and metadata_json columns.
    """
    m = doc.metadata
    resource_type = m["doc_type"]
    source_id = m["doc_id"]
    extra = _DEPRECATED_RESOURCE_EXTRA[resource_type]
    event_date = m.get(extra["event_date_field"])

    return (
        PatientDatabaseManager.get_stable_chunk_id(resource_type, source_id),
        source_id,
        resource_type,
        m["patient_id"],
        m.get("encounter_id"),
        m.get("code"),
        event_date,
        m.get("source_file"),
        doc.page_content,
        embedding_model,
        metadata_json_from_document(doc),
    )


def rows_to_documents(
    rows: Sequence[Mapping[str, Any]],
    resource_type: str,
    label: str,
) -> list[Document]:
    """
    DEPRECATED: Convert clinical rows to LangChain Documents.

    Args:
        rows: Table rows to wrap.
        resource_type: Resource key for row_to_document.
        label: Log label for the batch.

    Return:
        List of Documents (skips rows that raise during conversion).
    """
    documents: list[Document] = []
    print(f"Processing {len(rows)} {label}")
    for row in rows:
        try:
            documents.append(row_to_document(row, resource_type))
        except Exception as e:
            print(f"Error processing {label}: {e}")
    print(f"Processed {len(documents)} {label}")
    return documents
