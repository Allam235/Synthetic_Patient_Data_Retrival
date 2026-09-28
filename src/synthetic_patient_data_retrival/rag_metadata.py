"""
Single source of truth for RAG Document metadata and rag_chunks column mapping.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from langchain_core.documents import Document

from synthetic_patient_data_retrival.loadGeneratorData import PatientDatabaseManager

# ---------------------------------------------------------------------------
# Common metadata (every resource type uses doc_id / doc_type; others when applicable)
# ---------------------------------------------------------------------------

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

# Document.metadata key -> rag_chunks column (event_date uses per-resource event_date_field)
RAG_CHUNK_COLUMN_FROM_METADATA: Dict[str, str] = {
    "doc_id": "source_id",
    "doc_type": "resource_type",
    "patient_id": "patient_id",
    "encounter_id": "encounter_id",
    "code": "code",
    "source_file": "source_file",
}

# ---------------------------------------------------------------------------
# Per-resource config: row keys, Document metadata, page_content, JSON extras
# ---------------------------------------------------------------------------

RAG_RESOURCE_METADATA: Dict[str, Dict[str, Any]] = {
    "patient": {
        "description": "One index row per patients table row.",
        "doc_id_field": "patient_id",
        "event_date_field": "birth_date",
        "page_content": lambda row: (
            f"{row['first_name']} | {row['last_name']} | {row['gender']} | {row['birth_date']}"
        ),
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
        "description": "One index row per encounters table row.",
        "doc_id_field": "encounter_id",
        "event_date_field": "encounter_date",
        "page_content": lambda row: (
            f"{row['encounter_type']} | {row['reason']}"
        ),
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
        "description": "One index row per conditions table row.",
        "doc_id_field": "condition_id",
        "event_date_field": "onset_date",
        "page_content": lambda row: f"{row['description']}",
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
        "description": "One index row per observations table row.",
        "doc_id_field": "observation_id",
        "event_date_field": "observation_date",
        "page_content": lambda row: (
            f"{row['description']}: {row['value']} {row['unit']}"
        ),
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
        "description": "One index row per medications table row.",
        "doc_id_field": "medication_id",
        "event_date_field": "start_date",
        "page_content": lambda row: (
            f"{row['description']} | {row['start_date']} | {row['end_date']}"
        ),
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
        "description": "One index row per procedures table row.",
        "doc_id_field": "procedure_id",
        "event_date_field": "procedure_date",
        "page_content": lambda row: f"{row['description']}",
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
    config = RAG_RESOURCE_METADATA[resource_type]
    metadata: Dict[str, Any] = {"doc_type": resource_type}
    for meta_key, row_key in config["metadata_fields"].items():
        metadata[meta_key] = row[row_key]
    return metadata


def row_to_document(row: Mapping[str, Any], resource_type: str) -> Document:
    config = RAG_RESOURCE_METADATA[resource_type]
    return Document(
        page_content=config["page_content"](row),
        metadata=build_document_metadata(row, resource_type),
    )


def metadata_json_from_document(doc: Document) -> str:
    resource_type = doc.metadata["doc_type"]
    fields: Sequence[str] = RAG_RESOURCE_METADATA[resource_type]["metadata_json_fields"]
    payload = {
        key: doc.metadata[key]
        for key in fields
        if doc.metadata.get(key) is not None
    }
    return json.dumps(payload)


def document_to_rag_chunk_tuple(doc: Document, embedding_model: str) -> Tuple[Any, ...]:
    m = doc.metadata
    resource_type = m["doc_type"]
    source_id = m["doc_id"]
    config = RAG_RESOURCE_METADATA[resource_type]
    event_date = m.get(config["event_date_field"])

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
    documents: list[Document] = []
    print(f"Processing {len(rows)} {label}")
    for row in rows:
        try:
            documents.append(row_to_document(row, resource_type))
        except Exception as e:
            print(f"Error processing {label}: {e}")
    print(f"Processed {len(documents)} {label}")
    return documents
