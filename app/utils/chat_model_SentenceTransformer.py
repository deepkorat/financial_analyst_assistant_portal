# NLP and deep learning models for Q&A

import os
from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain_community.vectorstores import Chroma

from langchain.chains.question_answering import load_qa_chain


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
    '''
    Loads the pre-trained SentenceTransformer embeddings from Hugging Face.
    '''
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  # This model is free and works well for embedding
        return model
    except Exception as e:
        print("Something is wrong in the Embedding section: ", e)


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


## In this function -- many problems occur.
def final_call():
     # 1. Read PDF and extract text
    text = read_pdf("uploads/tcs.pdf")

    # 2. Split text into chunks
    docs = text_splitter(text)
    
    # 3. Create embeddings using SentenceTransformer
    model = embeddings()

     # 4. Create vector database
    doc_search = docsearch(docs, model)

    # 5. Create chain for QA
    chain = create_chain()
    return chain, docs


# Usage
if __name__ == "__main__":
     chain, docs = final_call()
     answer = question_answer(chain, "Who is CEO of the compnay", docs)
     print(answer)


    

