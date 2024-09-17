# app/routes.py
from flask import Blueprint, render_template

# Create a Blueprint
main = Blueprint('main', __name__)

@main.route('/')
def home():
    return "Welcome to the Financial Analyst Assistant Portal!"

@main.route('/upload')
def upload():
    return render_template('upload.html')

@main.route('/news')
def news():
    return "Company-specific news will appear here."
