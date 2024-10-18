import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    all_text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        all_text += page.extract_text()
    return all_text

# Function to process the report and answer user's question
def answer_question(text, question):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Use TF-IDF to find the most relevant sentence to the question
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between question and each sentence
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Find the index of the most similar sentence
    best_sentence_index = cosine_similarities.argmax()
    best_sentence = sentences[best_sentence_index]

    return best_sentence

# Streamlit app
st.title("Annual Report Question-Answering System")

# Step 1: Upload the annual report PDF
uploaded_pdf = st.file_uploader("Upload a company's annual report (PDF)", type="pdf")

if uploaded_pdf is not None:
    # Extract text from the uploaded PDF
    with st.spinner("Extracting text from the PDF..."):
        report_text = extract_text_from_pdf(uploaded_pdf)
        st.success("Text extracted from the report!")

    # Step 2: Ask a question
    question = st.text_input("Ask a question related to the annual report:")

    # Step 3: Provide the answer
    if question:
        with st.spinner("Finding the answer..."):
            answer = answer_question(report_text, question)
            st.write(f"Answer: {answer}")
