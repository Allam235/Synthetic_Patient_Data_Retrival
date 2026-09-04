import sqlite3
import json
from pathlib import Path

# Moves up 2 levels to project folder

DATA_PATH = Path(__file__).resolve().parents[2] / "data"
GEN_OUTPUT_PATH = DATA_PATH / "generator" / "output" / "fhir"
DB_PATH = DATA_PATH / "sqlite" / "patient.db"


def get_code_and_description(codeable):
    code = None
    description = None

    if codeable is None:
        return code, description

    codings = codeable.get("coding", [])
    if len(codings) > 0:
        coding = codings[0]
        code = coding.get("code")
        description = coding.get("display")

    if description is None:
        description = codeable.get("text")

    return code, description


def get_reference_id(reference):
    if reference is None:
        return None
    if ":" in reference:
        return reference.split(":")[-1]
    if "/" in reference:
        return reference.split("/")[-1]
    return reference


def get_observation_value(resource):
    value = None
    unit = None

    if "valueQuantity" in resource:
        quantity = resource["valueQuantity"]
        value = quantity.get("value")
        unit = quantity.get("unit")
    elif "valueString" in resource:
        value = resource["valueString"]
    elif "valueCodeableConcept" in resource:
        value = resource["valueCodeableConcept"].get("text")
    elif "valueBoolean" in resource:
        value = resource["valueBoolean"]

    return value, unit


def empty_patient_data():
    return {
        "patients": [],
        "encounters": [],
        "conditions": [],
        "observations": [],
        "medications": [],
        "procedures": [],
    }


def append_row(table_rows, parsed_row):
    headers, row = parsed_row
    if len(table_rows) == 0:
        table_rows.append(headers)
    table_rows.append(row)


def parse_patient(resource, source_file):
    names = resource.get("name", [])
    if len(names) == 0:
        name = {}
    else:
        name = names[0]

    given_names = name.get("given", [])
    first_name = " ".join(given_names)

    headers = ("id", "first_name", "last_name", "birth_date", "gender", "source_file")
    row = (
        resource["id"],
        first_name,
        name.get("family"),
        resource.get("birthDate"),
        resource.get("gender"),
        source_file,
    )
    return headers, row


def parse_encounter(resource, source_file):
    encounter_type = None
    encounter_types = resource.get("type", [])
    if len(encounter_types) > 0:
        code, encounter_type = get_code_and_description(encounter_types[0])

    reason = None
    reason_codes = resource.get("reasonCode", [])
    if len(reason_codes) > 0:
        code, reason = get_code_and_description(reason_codes[0])

    period = resource.get("period", {})
    encounter_date = period.get("start")

    subject = resource.get("subject", {})
    patient_id = get_reference_id(subject.get("reference"))

    headers = ("encounter_id", "patient_id", "encounter_date", "encounter_type", "reason", "source_file")
    row = (
        resource["id"],
        patient_id,
        encounter_date,
        encounter_type,
        reason,
        source_file,
    )
    return headers, row


def parse_condition(resource, source_file):
    code, description = get_code_and_description(resource.get("code"))

    subject = resource.get("subject", {})
    patient_id = get_reference_id(subject.get("reference"))

    encounter = resource.get("encounter", {})
    encounter_id = get_reference_id(encounter.get("reference"))

    onset_date = resource.get("onsetDateTime")
    if onset_date is None:
        onset_period = resource.get("onsetPeriod", {})
        onset_date = onset_period.get("start")

    headers = ("id", "patient_id", "encounter_id", "code", "description", "onset_date", "source_file")
    row = (
        resource["id"],
        patient_id,
        encounter_id,
        code,
        description,
        onset_date,
        source_file,
    )
    return headers, row


def parse_observation(resource, source_file):
    code, description = get_code_and_description(resource.get("code"))
    value, unit = get_observation_value(resource)

    subject = resource.get("subject", {})
    patient_id = get_reference_id(subject.get("reference"))

    encounter = resource.get("encounter", {})
    encounter_id = get_reference_id(encounter.get("reference"))

    observation_date = resource.get("effectiveDateTime")
    if observation_date is None:
        effective_period = resource.get("effectivePeriod", {})
        observation_date = effective_period.get("start")

    value_text = None
    if value is not None:
        value_text = str(value)

    headers = (
        "id",
        "patient_id",
        "encounter_id",
        "observation_date",
        "code",
        "description",
        "value",
        "unit",
        "source_file",
    )
    row = (
        resource["id"],
        patient_id,
        encounter_id,
        observation_date,
        code,
        description,
        value_text,
        unit,
        source_file,
    )
    return headers, row


def parse_medication(resource, source_file):
    code, description = get_code_and_description(resource.get("medicationCodeableConcept"))

    subject = resource.get("subject", {})
    patient_id = get_reference_id(subject.get("reference"))

    encounter = resource.get("encounter", {})
    encounter_id = get_reference_id(encounter.get("reference"))

    dispense_request = resource.get("dispenseRequest", {})
    validity_period = dispense_request.get("validityPeriod", {})
    end_date = validity_period.get("end")

    headers = ("id", "patient_id", "encounter_id", "description", "start_date", "end_date", "source_file")
    row = (
        resource["id"],
        patient_id,
        encounter_id,
        description,
        resource.get("authoredOn"),
        end_date,
        source_file,
    )
    return headers, row


def parse_procedure(resource, source_file):
    code, description = get_code_and_description(resource.get("code"))

    subject = resource.get("subject", {})
    patient_id = get_reference_id(subject.get("reference"))

    encounter = resource.get("encounter", {})
    encounter_id = get_reference_id(encounter.get("reference"))

    procedure_date = resource.get("performedDateTime")
    if procedure_date is None:
        performed_period = resource.get("performedPeriod", {})
        procedure_date = performed_period.get("start")

    headers = ("id", "patient_id", "encounter_id", "description", "procedure_date", "source_file")
    row = (
        resource["id"],
        patient_id,
        encounter_id,
        description,
        procedure_date,
        source_file,
    )
    return headers, row




def read_generator_output(output_path=None, patients=10):
    output_path = output_path or GEN_OUTPUT_PATH
    patient_files = []


    for file_path in output_path.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as file:
            bundle = json.load(file)
        patient_files.append((bundle, file_path.name))
        if len(patient_files) >= patients:
            break
    return patient_files


def merge_patient_data(all_data, patient_data):
    """
    
    """
    for key in all_data:
        rows = patient_data[key]
        if len(rows) == 0:
            continue
        if len(all_data[key]) == 0:
            all_data[key].extend(rows)
        else:
            all_data[key].extend(rows[1:])

def bundle_to_dict(bundle, source_file):
    """
    Converts a FHIR Bundle into a dictionary containing the row data for each resource type
    Args:
        bundle: FHIR Bundle
        source_file: Source file name
    Returns:
        data: Dictionary containing the row data for each resource type
    """
    data = empty_patient_data()

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Patient":
            append_row(data["patients"], parse_patient(resource, source_file))
        elif resource_type == "Encounter":
            append_row(data["encounters"], parse_encounter(resource, source_file))
        elif resource_type == "Condition":
            append_row(data["conditions"], parse_condition(resource, source_file))
        elif resource_type == "Observation":
            append_row(data["observations"], parse_observation(resource, source_file))
        elif resource_type == "MedicationRequest":
            append_row(data["medications"], parse_medication(resource, source_file))
        elif resource_type == "Procedure":
            append_row(data["procedures"], parse_procedure(resource, source_file))

    return data

def process_generator_output(patientsCount=10):
    patient_files = read_generator_output(patients=patientsCount)

    for bundle, source_file in patient_files:
        all_data = bundle_to_dict(bundle, source_file)
    return all_data


def create_tables(cursor):
    cursor.executescript(
        """

        DROP TABLE IF EXISTS patients;
        DROP TABLE IF EXISTS encounters;
        DROP TABLE IF EXISTS conditions;
        DROP TABLE IF EXISTS observations;
        DROP TABLE IF EXISTS medications;
        DROP TABLE IF EXISTS procedures;
        DROP TABLE IF EXISTS rag_audit;
        
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            birth_date TEXT,
            gender TEXT,
            source_file TEXT
        );

        CREATE TABLE encounters (
            encounter_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_date TEXT,
            encounter_type TEXT,
            reason TEXT,
            source_file TEXT,

            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE TABLE conditions (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_id TEXT,
            code TEXT,
            description TEXT,
            onset_date TEXT,
            source_file TEXT,

            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        );

        CREATE TABLE observations (
            observation_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_id TEXT,
            observation_date TEXT,
            code TEXT,
            description TEXT,
            value TEXT,
            unit TEXT,
            source_file TEXT,

            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        );

        CREATE TABLE medications (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_id TEXT,
            description TEXT,
            start_date TEXT,
            end_date TEXT,
            source_file TEXT,

            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        );

        CREATE TABLE procedures (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_id TEXT,
            description TEXT,
            procedure_date TEXT,
            source_file TEXT,

            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        );

        CREATE TABLE rag_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            patient_id TEXT,

            question TEXT NOT NULL,

            retrieved_chunks TEXT,

            context_sent_to_llm TEXT,

            answer TEXT,

            model TEXT,

            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );
        
        """
    )


def load_patients_data(cursor, patients):
    sql_statement = """
        INSERT INTO patients 
            (patient_id, first_name, last_name, birth_date, gender, source_file) 
        VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_statement, patients[1:])

def load_encounters_data(cursor, encounters):
    sql_statement = """
        INSERT INTO encounters 
            (encounter_id, patient_id, encounter_date, encounter_type, reason, source_file) 
        VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_statement, encounters[1:])

    
def load_medications_data(cursor, medications):
    sql_statement = """
        INSERT INTO medications 
            (id, patient_id, encounter_id, description, start_date, end_date, source_file) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_statement, medications[1:])


def load_conditions_data(cursor, conditions):
    sql_statement = """
        INSERT INTO conditions 
            (id, patient_id, encounter_id, code, description, onset_date, source_file) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_statement, conditions[1:])

def load_observations_data(cursor, observations):
    sql_statement = """
        INSERT INTO observations 
            (observation_id, patient_id, encounter_id, observation_date, code, description, value, unit, source_file) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_statement, observations[1:])

def load_procedures_data(cursor, procedures):
    sql_statement = """
        INSERT INTO procedures 
            (procedure_id, patient_id, encounter_id, description, procedure_date, source_file) 
        VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_statement, procedures[1:])

def load_patient_data(patient_data):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        load_patients_data(cursor, patient_data["patients"])
        load_medications_data(cursor, patient_data["medications"])
        load_encounters_data(cursor, patient_data["encounters"])
        load_conditions_data(cursor, patient_data["conditions"])
        load_observations_data(cursor, patient_data["observations"])
        load_procedures_data(cursor, patient_data["procedures"])
        conn.commit()

def intialize_database():
    """
        patients: number of patients from generator wanted
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        create_tables(cursor)

        conn.commit()

intialize_database()
data = process_generator_output(patientsCount=1)
load_patient_data(data)
