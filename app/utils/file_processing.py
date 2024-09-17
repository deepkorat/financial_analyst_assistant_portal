# Handling file uploads and document parsing

from PyPDF2 import PdfReader
import os
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

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
text = re.sub('[^a-zA-Z0-9]',' ', corpus)
text.lower()
text = word_tokenize(pattern)

lemmatizer = WordNetLemmatizer()
lemmatized_text = [lemmatizer.lemmatize(t) for t in text if t not in stopwords.words('english') ] 
lemmatized_text = ' '.join(lemmatized_text)





