# app/routes.py
import os
from flask import jsonify, request
from flask import Blueprint, render_template

# Create a Blueprint
main = Blueprint('main', __name__)

@main.route('/')
def home():
    # return "Welcome to the Financial Analyst Assistant Portal!"
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
            # return "Error: No selected file"

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        return render_template('dashboard.html', filename = file.filename )


    # return render_template('main.html', result = result)


@main.route('/ai-assist')
def ai_assist():
    # return render_template('ai-assist.html')
    return render_template('dashboard.html')


@main.route('/ai', methods=['POST'])
def ai_assist_form():
    query = request.form['query']
    answer="This is my annual report answer"
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
    response = "this is your anawer"
    
    # Send the response back to JavaScript as JSON
    return jsonify({'answer': response})


@main.route('/news')
def news():
    return "Company-specific news will appear here."
