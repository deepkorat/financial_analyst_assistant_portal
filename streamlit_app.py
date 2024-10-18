import streamlit as st
import PyPDF2

st.title("Upload Annual Report and Ask Questions")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()

    st.write("Extracted text from the PDF:")
    st.text(text)

    # Ask user for a question
    question = st.text_input("Ask a question:")
    if question:
        st.write(f"Your question: {question}")
        # Add logic to find the answer from the text
