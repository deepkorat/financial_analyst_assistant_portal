# NLP and deep learning models for Q&A
import sys
import os
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain_community.vectorstores import Chroma

from langchain.chains.question_answering import load_qa_chain

from config import OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI


## Add your API here. ##
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def read_pdf(path: str) -> str:
     '''
     This function take path and extract texts from Annual Report
     '''
     try:
          pdf_reader = PdfReader(path)
          my_text = []
          for page in pdf_reader.pages:
               text = page.extract_text()
               my_text.append(text)

          text = ' '.join(my_text)
          return text
     except Exception as e:
          print("Something is wrong in Path: ", e)

def text_splitter(text: str)->list:
     '''
     The function split the text into several chunks. It takes raw text of Annual report as a argument.
     Function returns Docs as list.  
     '''
     try:
          text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap  = 32, length_function = len,)
          docs = text_splitter.split_text(text)
          return docs
     except Exception as e:
          print("Sorry! Problem Occure in text splitter section: ", e)

def embeddings():
     try:
          embeddings = OpenAIEmbeddings() ## Download embeddings from OpenAI
          return embeddings
     except Exception as e:
          print("Something is wrong in Embedding section: ", e)
          
     
def docsearch(docs, embeddings):
     '''
     This function proceed to create a vector database (our knowledge base) using the FAISS library and the OpenAI embeddings.
     It takes two arguments --> 1. Docs,  2. embeddings
     ''' 
     try:
          docsearch = FAISS.from_texts(docs, embeddings)
          return docsearch
     except Exception as e:
          print("Sorry, We are unable to create FAISS Vector Database: ", e)


def create_chain():
     '''
     Create chain using openAI.
     '''
     chain = load_qa_chain(OpenAI(), chain_type="stuff")


def question_answer(chain, query, docs):
     '''
     This function used for question answer module.
     It takes 3 parameter.
     1. chain - OpenAI chain 
     2. query - Enter user query.
     3. docs - Vector database which you creat by FAISS.
     '''
     query = "Which company's annual report it is?"
     docs = docsearch.similarity_search(query)
     chain.run(input_documents=docs, question=query)




# Usage
if __name__ == "__main__":
    try:
        # 1. Read PDF and extract text
        pdf_path = "uploads/tcs.pdf"  # Replace with the actual path to the PDF
        text = read_pdf(pdf_path)
        if not text:
            raise ValueError("Failed to extract text from PDF.")

        # 2. Split text into chunks
        docs = text_splitter(text)
        if not docs:
            raise ValueError("Text splitting failed.")

        # 3. Get embeddings
        embeddings_model = embeddings()
        if not embeddings_model:
            raise ValueError("Failed to load embeddings.")

        # 4. Create vector database
        vector_db = docsearch(docs, embeddings_model)
        if not vector_db:
            raise ValueError("Failed to create FAISS vector database.")

        # 5. Create chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        if not chain:
            raise ValueError("Failed to create OpenAI QA chain.")

        # 6. Ask question and get answer
        user_query = "Which company's annual report is it?"
        related_docs = vector_db.similarity_search(user_query)
        if not related_docs:
            raise ValueError("No relevant documents found for the query.")

        answer = chain.run(input_documents=related_docs, question=user_query)
        print("Answer:", answer)

    except Exception as e:
        print("An error occurred:", e)


