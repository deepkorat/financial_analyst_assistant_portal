# NLP and deep learning models for Q&A

import os
from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain_community.vectorstores import Chroma

from langchain.chains.question_answering import load_qa_chain

# import openai
from langchain_huggingface import HuggingFaceEndpoint


# from langchain_community.llms import OpenAI



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
     llm = HuggingFaceEndpoint(endpoint_url="HUGGINGFACE_API_KEY", api_key = "HUGGINGFACE_API_KEY")
     # llm = HuggingFaceEndpoint(endpoint_url="google/flan-t5-small", api_key="")
     chain = load_qa_chain(llm, chain_type="stuff")


     # chain = load_qa_chain(OpenAI(), chain_type="stuff")


def question_answer(chain, query, docs ):
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
    print("Text reading is done")

    # 2. Split text into chunks
    docs = text_splitter(text)
    print("text splitting is done")
    
    # 3. Create embeddings using SentenceTransformer
    model = embeddings()
    print("Model is done")

     # 4. Create vector database
    doc_search = docsearch(docs, model)
    print("doc_search is done")

    # 5. Create chain for QA
    chain = create_chain()
    print("all done")
    return chain, docs


# Usage
if __name__ == "__main__":
     modelTrained = False
     if modelTrained == False:
          chain, docs = final_call()
          answer = question_answer(chain, "Who is CEO of the compnay", docs)
          print(answer)



    

