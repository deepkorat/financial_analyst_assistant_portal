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
    return render_template('uploadtry.html')


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
        return "File successfully save"


    # return render_template('main.html', result = result)

@main.route('/news')
def news():
    return "Company-specific news will appear here."
