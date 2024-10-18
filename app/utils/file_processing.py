# Handling file uploads and document parsing

from PyPDF2 import PdfReader
import os
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# libraries for model Word2Vec
import gensim
from gensim.models import Word2Vec

## Define the path to the uploads folder
current_dir = os.path.dirname(__file__)
uploads_dir = os.path.join(current_dir, '..\..', 'uploads')  # Adjust path if needed based on the notebook's location
pdf_file = 'tcs.pdf'
pdf_path = os.path.join(uploads_dir, pdf_file)
print(pdf_path)

reader = PdfReader(pdf_path)

## extrat entire text from pdf
corpus = ""
for page in reader.pages:
     corpus+=page.extract_text()

## Text Preprocessing
sent_tokenized_data = sent_tokenize(corpus)
sent_tokenized_data

preprocessed_data = []
for sent in sent_tokenized_data:
     text = re.sub('[^a-zA-Z0-9]',' ', sent)
     text = text.lower()
     text = word_tokenize(text)
     lemmatizer = WordNetLemmatizer()
     lemmatized_text = [lemmatizer.lemmatize(t) for t in text if t not in stopwords.words('english') ]
     # lemmatized_text = ' '.join(lemmatized_text)
     preprocessed_data.append(lemmatized_text)

## Word Embedding
# Word2Vec
model=Word2Vec(preprocessed_data,window=5,min_count=2)
print("\n\n\n_______All__Unique__Words________\n\n\n", model.wv.index_to_key)
print("\n\n\n_______Model__Count________\n\n\n", model.corpus_count)
print("\n\n\n________Model__Similarity__By__Word________\n\n\n", model.wv.similar_by_word('ceo'))





