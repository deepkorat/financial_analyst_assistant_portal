import os
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

def read_pdf(path: str) -> str:
    '''Extracts text from a PDF file.'''
    try:
        pdf_reader = PdfReader(path)
        my_text = [page.extract_text() for page in pdf_reader.pages]
        return ' '.join(my_text)
    except Exception as e:
        print("Error reading PDF: ", e)

def text_splitter(text: str) -> list:
    '''Splits text into chunks and saves them.'''
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        return text_splitter.split_text(text)
    except Exception as e:
        print("Error in text splitter: ", e)

def save_split_text(docs, filename="data/split_text.json"):
    '''Saves split text chunks to a JSON file.'''
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(docs, f)
    print("Text chunks saved to:", filename)

def load_split_text(filename="data/split_text.json") -> list:
    '''Loads split text chunks from a JSON file.'''
    with open(filename, "r") as f:
        return json.load(f)

def load_embeddings_model():
    '''Loads pre-trained SentenceTransformer embeddings.'''
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print("Error loading embeddings model: ", e)

def create_docsearch(docs, embedding_model):
    '''Creates a vector database using FAISS with precomputed embeddings.'''
    try:
        # Generate embeddings for each document chunk
        embeddings = [embedding_model.encode(doc) for doc in docs]
        
        # Create FAISS index with embeddings and documents
        docsearch = FAISS.from_embeddings(embeddings, [Document(page_content=doc) for doc in docs])
        
        print("FAISS Vector Database created successfully.")
        return docsearch
    except Exception as e:
        print("Error creating FAISS Vector Database: ", e)
        return None

def retrieve_answer(query, docsearch):
    '''Retrieves the most relevant documents for a given query.'''
    try:
        if docsearch is None:
            print("Docsearch is not initialized.")
            return None

        relevant_docs = docsearch.similarity_search(query)
        return relevant_docs[0].page_content if relevant_docs else "No relevant information found."
    except Exception as e:
        print("Error in retrieving answer: ", e)

# Main function with caching
def final_call():
    split_text_file = "data/split_text.json"
    
    if os.path.exists(split_text_file):
        print("Loading split text from cache.")
        docs = load_split_text(split_text_file)
    else:
        print("Reading and splitting text from PDF.")
        text = read_pdf("uploads/tcs.pdf")
        docs = text_splitter(text)
        save_split_text(docs, split_text_file)

    embedding_model = load_embeddings_model()
    doc_search = create_docsearch(docs, embedding_model)
    
    return doc_search

# Usage
if __name__ == "__main__":
    docsearch = final_call()
    question = "Who is the CEO of the company?"
    if docsearch:
        answer = retrieve_answer(question, docsearch)
        print("Answer:", answer)
    else:
        print("Could not retrieve an answer due to vector database creation failure.")
