# Unlocking the Potential of Retrieval-Augmented Generation (RAG) for Diagnosis in Healthcare
######################################################################################################################
# Key Components of RAG in this Application
# 1. Retrieval Component (Chromadb)
# There is knowledge base stored in a text file (parkinson_disease_knowledge.txt).
# Knowledge is embed using SentenceTransformer.
# The embeddings are stored in a Chromadb for efficient similarity search.
# When a query is generated based on gait data, relevant knowledge is retrieved using Chromadb.
# 2. Augmented Generation (Groq API with LLaMA-3)
# The retrieved knowledge is passed as context into a LLM (LLaMA-3.1-8b-instant via Groq).
# The LLM generates a diagnostic response, strictly based on the retrieved knowledge.
######################################################################################################################
# Retrieval: Uses Chromadb to find relevant knowledge from a structured knowledge base.
# Augmentation: Injects this retrieved knowledge into the LLM prompt.
# Generation: The LLM (LLaMA-3) generates a response only based on the retrieved knowledge, preventing hallucinations.
######################################################################################################################

from fastapi import FastAPI, HTTPException
from typing import Dict
from groq import Groq
import logging
import shutil
import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from dotenv import load_dotenv

# set logging to info level
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# groq api key
groq_api_key = os.getenv("groq_api_key")

# Initialize FastAPI app
app = FastAPI()

KNOWLEDGE_PATH = "knowledge_base"
CHROMA_PATH = "chroma"

# Function to oad the documents
def load_documents(DATA_PATH):
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

# Function for data chunking
def chunking(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, 
                                               chunk_overlap=500,
                                               length_function=len,
                                               add_start_index=True)
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

# Function to load the embedding model
def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Function to save the embedded text in ChromaDB
def save_to_chroma(chunks: list[Document], CHROMA_PATH, embedding_model):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=CHROMA_PATH
    )
    db.persist()
    logging.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Function to load the saved ChromaDB vector database
def load_chroma(CHROMA_PATH, embedding_model):
    # Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    return db

# Function to process gait parameters JSON
def extract_gait_features(data: Dict):
    gait_data = {}
    for param in data["result_details"]:
        name = param["name"]
        description = param["description"]
        score_dict = param["score_dict"]
        nominal_value = score_dict["nominal_value"] 
        nominal_lower = score_dict["nominal_lower"] 
        nominal_upper = score_dict["nominal_upper"]    
        unit = score_dict["unit"]        
        gait_data[name] = (nominal_value, nominal_lower, nominal_upper, unit, description)
    return gait_data

# Function to load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to generate a medical query based on gait features
def generate_query(gait_data):
    query = "Patient gait analysis results: "
    for key, value in gait_data.items():
        if isinstance(value, tuple):
            query += f"{key} for the patient is {value[0]} {value[3]} and the reference range for healthy patient is between {value[1]} {value[3]} and {value[2]} {value[3]}, "
    query += "What does this indicate about Parkinson's disease?"
    return query

# Function to retrieve relevant knowledge from ChromaDB
def retrieve_knowledge(query, db):
    
    results = db.similarity_search_with_relevance_scores(query, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    context_sources = [doc.metadata.get("source", None) for doc, _score in results]

    return context_text, context_sources

# Function to get the PROMPT_TEMPLATE based on query and retrieved knowledge
def get_prompt_template(query, retrieved_knowledge):
    
    PROMPT_TEMPLATE = f"""

    You are a specialized medical AI diagnostic assistant providing gait analysis diagnoses. 

    Answer the question based on following relevant medical insights from the knowledge base:

    {retrieved_knowledge}

    **Determine Relevance**
    - If the question does not pertain to gait analysis or is unrelated to the relevant medical insights, respond with: "Diagnose is not possible because it is out of scope." 
    - If uncertain, respond with: "Insufficient data!" 

    **Diagnostic Assessment**
    - Strictly base your responses on relevant medical insights from the knowledge base.
    - Provide a diagnostic assessment of the patient's gait and possible relation to Parkinson's disease based on the medical insights.

    Now, based on the above conditions, generate the response but don't repeat the determine relevance and diagnostic assessments. Do not use I and always answer as a third person.
    At the end diagnose should have the result saying if the user has parkinson disease or not with the word "likelihood".

    -----

    Answer the question based on the above context: {query}
    """

    return PROMPT_TEMPLATE

# Function to generate diagnosis using Grok and llama, based on PROMPT_TEMPLATE
def generate_diagnosis(PROMPT_TEMPLATE, api_key):

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
                                                model="llama-3.1-8b-instant",
                                                messages=[{"role": "system", "content": "You answer only based on the provided JSON data."},
                                                        {"role": "user", "content": PROMPT_TEMPLATE}],
                                                temperature=0,
                                                max_completion_tokens=1024,
                                                top_p=1,
                                                stream=True,
                                                stop=None,
                                            )
    
    return completion

# Load the documents from knowledge base
documents = load_documents(KNOWLEDGE_PATH)
logging.info("Documents are loaded successfully...")

# Data chunking
chunks = chunking(documents)
logging.info("Chunking the documents completed...")

# load the embedding model
embedding_model = load_embedding_model()
logging.info("Embedding model loaded...")

# Save the embed text into ChromaDB vector database if not available in the dir
if not os.path.isdir(CHROMA_PATH):
    save_to_chroma(chunks, CHROMA_PATH, embedding_model)
    logging.info("Generated the ChromaDB vector database and saved in the dir...")

# FastAPI Endpoint
@app.post("/diagnose_parkinson/")
async def analyze_gait(data: Dict):
    try:
        
        # Extract gait features
        gait_data = extract_gait_features(data)
        logging.info("Gait data extracted...")

        # Generate query based on gait data
        query = generate_query(gait_data)
        query += "What does this indicate about Parkinson's disease?"
        logging.info("Query generated...")

        # load the vector_store db
        vector_store = load_chroma(CHROMA_PATH=CHROMA_PATH, embedding_model=embedding_model)
        logging.info("Vector databased ChromaDB loaded from the local dir...")

        # Retrieve relevant knowledge
        retrieved_knowledge, retrieved_sources = retrieve_knowledge(query, vector_store)
        logging.info("Relevant knowledge retrived based on the query...")

        # get prompt template
        PROMPT_TEMPLATE = get_prompt_template(query=query, retrieved_knowledge=retrieved_knowledge)
        logging.info("Prompt template generated...")

        # Generate diagnosis using GPT
        diagnosis = generate_diagnosis(PROMPT_TEMPLATE, groq_api_key)
        diagnosis_text = "".join([chunk.choices[0].delta.content or "" for chunk in diagnosis])
        diagnosis_text += " However, possible further assessment with the doctors is recommended and diagnosis should be confirmed by a qualified medical professional."
        logging.info("Diagnosis generated...")

        payload =  {
                "query": query,
                "retrieved_sources": retrieved_sources,
                "retrived_content": retrieved_knowledge,
                "diagnosis": diagnosis_text.replace("\n", " ")

            }
        
        return payload["diagnosis"]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))