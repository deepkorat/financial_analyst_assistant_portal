import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
# nltk.download('punkt')

# Load the text of the document (use uploaded file's content)
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

# print(document)

# Split document into sentences
sentences = nltk.sent_tokenize(document)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

def answer_question(question, sentences, tfidf_matrix):
    """
    Find the most relevant answer from the document based on a question.
    """
    # Transform the question to the same vector space
    question_tfidf = vectorizer.transform([question])
    # Compute similarity scores
    similarities = cosine_similarity(question_tfidf, tfidf_matrix)
    # Find the most relevant sentence
    most_relevant_idx = similarities.argsort()[0][-1]
    return sentences[most_relevant_idx]

# Example Usage
question = "Dividend yield of current year?"
answer = answer_question(question, sentences, tfidf_matrix)
print("Answer:", answer)
