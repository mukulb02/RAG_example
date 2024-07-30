from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = FastAPI()

# Constants for Chroma settings
CHROMA_SETTINGS = {"index_type": "flat", "metric_type": "l2"}
similarity_threshold = 1.5

# Initialize Chroma client and create or get a collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection")

# Define your PDF files and initialize necessary components
pdf_files = ["WhatYouNeedToKnowAboutWOMENSHEALTH.pdf", "Clinical Manual of Women's Mental Health.pdf"]
for pdf_file in pdf_files:
    if not os.path.isfile(pdf_file):
        raise ValueError(f"PDF file '{pdf_file}' not found.")
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    texts = [doc.page_content for doc in documents]
    collection.upsert(documents=texts, ids=[f"{pdf_file}_{i}" for i in range(len(documents))])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.1")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.1")

def hf_wrapper(inputs):
    question = inputs["question"]
    response = model.generate(
        tokenizer.encode(question, return_tensors='pt'),
        max_new_tokens=90,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(response[0], skip_special_tokens=True)
    return {"answer": answer}

def truncate_context(context, max_words=30):
    words = context.split()
    if len(words) > max_words:
        truncated_context = " ".join(words[:max_words])
    else:
        truncated_context = context
    return truncated_context

def chat(query, chat_history=[]):
    results = collection.query(query_texts=[query], n_results=2)
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadata_list = results.get('metadatas', [[]])[0]
    context = " ".join(documents)
    truncated_context = truncate_context(context, max_words=30)
    sources = [meta.get('source', 'Unknown') if meta else 'Unknown' for meta in metadata_list]
    source_info = f"Source: {', '.join(sources)}"
    response_text_with_rag = ""
    if distances and distances[0] < similarity_threshold:
        prompt_with_context = f"Context Page: {truncated_context}\n{source_info}\n\nQuestion: {query}\nAnswer:"
        response_with_context = hf_wrapper({"question": prompt_with_context})
        response_text_with_rag = response_with_context.get('answer', 'No answer generated.')
    else:
        response_text_with_rag = "The answer to your query is not found in the provided documents."
    combined_response = f"Answer with document context: {response_text_with_rag}\n{source_info}"
    chat_history.append((query, combined_response))
    return combined_response, chat_history

class Query(BaseModel):
    query: str

@app.get("/predict")
def predict(query: str):
    try:
        response, _ = chat(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
