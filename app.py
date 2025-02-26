import streamlit as st
from pdf_processor import process_pdf
from embeddings import create_vector_store
from translation import translate
from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Streamlit UI
st.title("📄 Nepali PDF Chatbot")
st.write("Upload a PDF in Nepali and chat with it in English or Nepali!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    if pdf_file:
        with st.spinner("Processing PDF..."):
            # Extract text and create vector store
            texts = process_pdf(pdf_file)
            st.session_state.vector_store = create_vector_store(texts)
            st.success("PDF processed successfully!")

# Language selection
response_language = st.radio("Response Language", ["English", "Nepali"])

# Initialize Hugging Face pipeline for text generation
generator = pipeline("text-generation", model="gpt2")  # You can replace "gpt2" with other models

# Chat interface
user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append(("user", user_input))

    try:
        # Translate query to Nepali
        nepali_query = translate(user_input, src='en', dest='ne')

        # Retrieve relevant chunks from the PDF
        docs = st.session_state.vector_store.similarity_search(nepali_query, k=3)
        english_context = "\n".join([translate(doc.page_content) for doc in docs])

        # Generate response using Hugging Face
        prompt = f"Context: {english_context}\nQuestion: {user_input}\nAnswer:"
        response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

        # Translate response if needed
        if response_language == "Nepali":
            response = translate(response, src='en', dest='ne')

        # Add bot response to chat history
        st.session_state.chat_history.append(("bot", response))

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.chat_history.append(("bot", "Sorry, something went wrong. Please try again."))

# Display chat history
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)