# app/routes.py

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
from PyPDF2 import PdfReader

from flask import jsonify, request
from flask import Blueprint, render_template
from app.utils.chat_model_SentenceTransformer import final_call
# from utils.chat_model_SentenceTransformer import read_pdf, text_splitter, embeddings, docsearch, question_answer


##### OPEN AI MODEL LIBRARIES START######
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv("config.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# from config import OPENAI_API_KEY
#### OPEN AI MODEL LIBRARES END #####



# CREATE a BLUEPRINT
main = Blueprint('main', __name__)



###### OPEN AI MODEL FUNCTIONS START ######
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

###### OPEN AI MODEL FUNCTIONS END ######







@main.route('/')
def home():
    return render_template('index.html')

@main.route('/upload_page')
def upload_page():
    return render_template('base.html')

@main.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':    
        if 'annual_report' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['annual_report']
        if file.filename == '':
            return jsonify({"error": "No selected file"})

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # 1. Read PDF and extract text
        # text = read_pdf(file_path)

        # # 2. Split text into chunks
        # docs = text_splitter(text)
        
        # # 3. Create embeddings using SentenceTransformer
        # model = embeddings()
        # print(model)
    
        return render_template('dashboard.html', filename = file.filename )

    # return render_template('main.html', result = result)


@main.route('/ai-assist')
def ai_assist():
    # return render_template('ai-assist.html')
    return render_template('dashboard.html')

@main.route('/gpt_dashboard')
def gpt_dashboard():
    return render_template('gpt_dashboard.html')

@main.route('/dashboard2')
def dashboard2():
    return render_template('dashboard2.html')



nltk.download('punkt')

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

document = read_pdf("uploads/tcs.pdf")


## FOR TF IDF BASED MODEL
sentences = nltk.sent_tokenize(document)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

def TF_IDF_answer_question(question):
    question_tfidf = vectorizer.transform([question])
    similarities = cosine_similarity(question_tfidf, tfidf_matrix)
    most_relevant_idx = similarities.argsort()[0][-1]
    return sentences[most_relevant_idx]



@main.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    answer = TF_IDF_answer_question(question)

    response_data = {"message": "Data received successfully", "response": answer}
    return jsonify(response_data)
    # return jsonify({'question': question, 'answer': answer})





