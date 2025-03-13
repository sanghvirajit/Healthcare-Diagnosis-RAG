# Healthcare diagnosis with RAG
Enhancing Healthcare Diagnostics with Retrieval-Augmented Generation (RAG): Leveraging LangChain, ChromaDB, and Hugging Face

## Clone the github repository
```sh
git clone https://github.com/sanghvirajit/Healthcare_diagnosis_RAG.git
```

## Create a virtual environment
```sh
conda create --name rag_env python=3.11
```

## Activate the virtual environment
```sh
conda activate rag_env
```

## Install the required libraries
```sh
pip install -r requirements.txt
```

## Run the local Flask API
```sh
uvicorn rag:app --host 0.0.0.0 --port 8000
```

## Send the request and get the response
```sh
python rag_response.py
```

