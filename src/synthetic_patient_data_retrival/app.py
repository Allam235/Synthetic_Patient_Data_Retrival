import streamlit as st
from synthetic_patient_data_retrival.loadGeneratorData import PatientDatabaseManager



st.set_page_config(
    page_title="Synthetic Patient Data Retrieval",
    page_icon="🩺",
)

st.title("Synthetic Patient Data Retrieval")
st.write("Ask a question about the synthetic patient data.")

with st.form("patient_query_form"):
    query = st.text_area(
        "Question",
        placeholder="Enter your question here...",
        height=150,
    )
    submitted = st.form_submit_button("Submit")

if submitted:
    if query.strip():
        st.subheader("Submitted question")
        st.write(query)
    else:
        st.warning("Enter a question before submitting.")


