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

# Create a Blueprint
main = Blueprint('main', __name__)

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



    
# @main.route('/ask', methods=['POST'])
# def ask_question():
#     # Get the question from the request (sent via JavaScript)
#     question = request.json.get('question')
    
#     # # Example simple model logic for answering
#     # chain, docs = final_call()
#     # answer = question_answer(chain, query, docs)
#     # # response = "this is my flask answer."
#     # response = answer
#     response = "This is my flask answer."
   
#     # Send the response back to JavaScript as JSON
#     response_data = {"message": "Data received successfully", "response": response}

#     return jsonify(response_data)


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




@main.route('/news')
def news():
    return "Company-specific news will appear here."
