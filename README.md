# Chatbot with LangChain and Streamlit  

This project is a chatbot application built using **LangChain** and **Streamlit**, designed to process PDFs and provide intelligent responses using embeddings and translation. It includes multiple components such as `app.py`, which serves as the main Streamlit application, `embeddings.py` for handling text embeddings, `pdf_processor.py` for extracting and processing text from PDF documents, and `translation.py` for language translation.  

## Steps to run this project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the chatbot:**
   ```bash
   streamlit run app.py
   ```

The chatbot provides features such as PDF text extraction, embedding-based responses, and language translation support. The project leverages **LangChain**, **Streamlit**, **Python**, and optionally the **OpenAI API**.
