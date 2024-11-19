from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import torch
from PyPDF2 import PdfReader


# Initialize Flask app
app = Flask(__name__)

# ======== Load BERT Model and Tokenizer (Once) ========
print("Loading BERT model and tokenizer...")
bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# ======== Load TF-IDF Model (Once) ========
print("Preparing TF-IDF model...")
nltk.download('punkt')

# Load document for both models (convert your PDF into text first)

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


# Tokenize document into sentences for TF-IDF
sentences = nltk.sent_tokenize(document)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

# ======== Define BERT Question-Answering Logic ========
def bert_answer(question, context):
    """
    Use BERT to find the answer to the question from the context.
    """
    inputs = bert_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = bert_tokenizer.convert_tokens_to_string(
        bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

# ======== Define TF-IDF Question-Answering Logic ========
def tfidf_answer(question, sentences, tfidf_matrix):
    """
    Use TF-IDF to find the most relevant sentence as the answer.
    """
    question_tfidf = tfidf_vectorizer.transform([question])
    similarities = cosine_similarity(question_tfidf, tfidf_matrix)
    most_relevant_idx = similarities.argsort()[0][-1]
    return sentences[most_relevant_idx]

# ======== Flask Routes ========
@app.route('/ask', methods=['POST'])
def ask():
    """
    Main endpoint to handle question-answering requests.
    """
    data = request.json
    question = data.get("question", "")
    model_type = data.get("model_type", "bert")  # Default to BERT if not specified

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Choose model based on request
    if model_type == "bert":
        answer = bert_answer(question, document)
    elif model_type == "tfidf":
        answer = tfidf_answer(question, sentences, tfidf_matrix)
    else:
        return jsonify({"error": "Invalid model_type. Use 'bert' or 'tfidf'"}), 400

    return jsonify({"question": question, "model": model_type, "answer": answer})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
