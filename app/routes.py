# app/routes.py
import os
from flask import jsonify, request
from flask import Blueprint, render_template
from app.utils.chat_model_SentenceTransformer import final_call
# from utils.chat_model_SentenceTransformer import read_pdf, text_splitter, embeddings, docsearch, question_answer

# Create a Blueprint
main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/main')
def my_main():
    return render_template('main.html')

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

@main.route('/ai', methods=['POST'])
def ai_assist_form():
    query = request.form['query']

    answer="This is my annual report answer" ## implement your model here and fetch answer
    return render_template('ai-assist.html', query= query, answer=answer)

@main.route('/dashboard2')
def dashboard2():
    return render_template('dashboard2.html')

@main.route('/data')
def send_data():
    data = {'key': 'value'}
    return jsonify(data)



    
@main.route('/ask', methods=['POST'])
def ask_question():
    # Get the question from the request (sent via JavaScript)
    question = request.json.get('question')
    
    # Example simple model logic for answering
    chain, docs = final_call()
    answer = question_answer(chain, query, docs)
    # response = "this is my flask answer."
    response = answer
   
    # Send the response back to JavaScript as JSON
    response_data = {"message": "Data received successfully", "response": response}

    return jsonify(response_data)



@main.route('/news')
def news():
    return "Company-specific news will appear here."
