# -*- coding: utf-8 -*-
"""RAG-main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WFF2ApwDFZlfDRg1Ag5WxHDfRR1yLXUG
"""

!pip install python-dotenv
!pip install streamlit
!pip install PyPDF2
!pip install langchain
!pip install faiss-GPU
!pip install -U langchain-community
!pip install pyngrok
!pip install transformers bitsandbytes
!pip install accelerate
!pip install chromadb
!pip install pypdf
!pip install sentence-transformers

# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
#from htmlTemplates import css, bot_template, user_template
import os
import chromadb
from langchain.document_loaders import PyPDFLoader
import re

# Constants for Chroma settings
CHROMA_SETTINGS = {"index_type": "flat", "metric_type": "l2"}
persist_directory = "db"
similarity_threshold = 1.5

# Initialize Chroma client
chroma_client = chromadb.Client()

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="my_collection")

# List of PDF files to process
pdf_files = ["WhatYouNeedToKnowAboutWOMENSHEALTH.pdf", "Clinical Manual of Women's Mental Health.pdf"]

# Iterate through each PDF file
for pdf_file in pdf_files:
    # Check if the PDF file exists
    if not os.path.isfile(pdf_file):
        raise ValueError(f"PDF file '{pdf_file}' not found.")

    print(f"Loading file: {pdf_file}")
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # Extract text from the document (assuming PyPDFLoader returns a list of documents)
    texts = [doc.page_content for doc in documents]

    # Add documents to the collection
    collection.upsert(documents=texts, ids=[f"{pdf_file}_{i}" for i in range(len(documents))])

print("All PDFs processed and added to the collection.")

print(documents)

# Function to clear the collection by deleting it and recreating it
#def clear_collection(client, collection_name):
    #client.delete_collection(collection_name)
    #return client.get_or_create_collection(name=collection_name)

# Clear the collection before adding new documents
#collection = clear_collection(chroma_client, "my_collection")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
content_for_embedding = [doc.page_content for doc in documents]
# Convert text to embeddings
embeddings = embeddings_model.embed_documents(content_for_embedding)

# Prepare metadata
metadatas = [{"source": pdf_file, "page_number": i + 1, "content": content} for i, content in enumerate(content_for_embedding)]

    # Prepare IDs
ids = [f"{pdf_file}_{i}" for i in range(len(documents))]

    # Add documents with embeddings to the collection
collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

print("All PDFs processed and added to the collection.")

class HuggingFacePipelineWrapper:
    def __init__(self, model, tokenizer, max_length=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, inputs):
        prompt = inputs["question"]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_length)  # Use max_new_tokens to avoid warning
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"answer": answer}

# Define the path to your fine-tuned model in Google Drive
#model_path = "/content/drive/MyDrive/TinyLLAMA_finetuned"
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.1")
model = AutoModelForCausalLM.from_pretrained(model_path)

# Initialize HuggingFacePipelineWrapper with model and tokenizer
hf_wrapper = HuggingFacePipelineWrapper(model, tokenizer, max_length=100)

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

# Function to truncate context to 30 words
def truncate_context(context, max_words=40):
    words = context.split()
    if len(words) > max_words:
        truncated_context = " ".join(words[:max_words])

    else:
        truncated_context = context
    return truncated_context

def chat(query, chat_history=[]):
    # Step 1: Retrieve context from documents
    results = collection.query(query_texts=[query], n_results=2)

    # Print results for debugging
    #print("Results:", results)

    # Extract the documents, distances, and metadata
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]
    metadata_list = results.get('metadatas', [[]])[0]

    # Combine the documents to form the context
    context = " ".join(documents)
    truncated_context = truncate_context(context, max_words=30)# Extract sources from metadata
    sources = [meta.get('source', 'Unknown') if meta else 'Unknown' for meta in metadata_list]
    source_info = f"Source: {', '.join(sources)}"

    response_text_with_rag = ""
    if distances and distances[0] < similarity_threshold:
        print(distances[0])
        prompt_with_context = f"Context Page: {truncated_context}\n{source_info}\n\nQuestion: {query}\nAnswer:"
        response_with_context = hf_wrapper({"question": prompt_with_context})
        response_text_with_rag = response_with_context.get('answer', 'No answer generated.')

        print("RAG-based response:", response_text_with_rag)
    else:
        response_text_with_rag = "The answer to your query is not found in the provided documents."


    combined_response = f"Answer with document context: {response_text_with_rag}\n{source_info}"

    chat_history.append((query, combined_response))
    return combined_response, chat_history

import torch
# Example interaction
query = "What is the function of fallopian tubes?"
#answer,
chat_history = chat(query)
#print("Combined Answer:", answer)

