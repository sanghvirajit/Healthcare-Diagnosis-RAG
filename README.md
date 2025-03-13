# Healthcare Diagnosis with RAG
Enhancing Healthcare Diagnostics with Retrieval-Augmented Generation (RAG): Leveraging LangChain, ChromaDB, and Hugging Face

# Introduction
The integration of AI in healthcare is revolutionising diagnostics, particularly with advancements in Retrieval-Augmented Generation (RAG). By combining retrieval-based knowledge with generative capabilities, RAG enhances traditional AI models, delivering more accurate and reliable results. In this blog, we explore how RAG can be applied to diagnosing Parkinson's disease using gait analysis. Leveraging LangChain, ChromaDB vector database for efficient data retrieval and LLaMA-3 via the Groq API for intelligent generation, we demonstrate a cutting-edge approach to AI-driven medical diagnostics.

LangChain is a popular open-source framework designed for developing applications powered by Large Language Models (LLMs). It provides tools and abstractions to integrate LLMs with external data sources, memory, and reasoning capabilities, making it easier to build AI-driven applications such as chatbots, RAG (Retrieval-Augmented Generation) systems, and autonomous agents.

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

