import os
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import glob
import re
from fuzzywuzzy import fuzz

# Load environment variables (Google API key)
load_dotenv()

# Configure the Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define paths to timetable images for different branches
timetable_paths = {
    "aiml-a": "data/timetables/aiml-a.jpg",
    "aiml-b": "data/timetables/aiml-b.jpg",
    "aids-a": "data/timetables/aids-a.jpg",
    "aids-b": "data/timetables/aids-b.jpg",
    "iot": "data/timetables/iot.jpg"
}

# Function to preprocess user question for common variations
def preprocess_question(user_question):
    # Convert "vips" to "vips-tc" if detected
    if re.search(r"\bvips\b", user_question, re.IGNORECASE):
        user_question = re.sub(r"\bvips\b", "vips-tc", user_question, flags=re.IGNORECASE)
    return user_question

# Function to extract text from multiple PDF files
def get_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to initialize and store FAISS vector store with multiple documents
def initialize_vector_store(pdf_folder="data/pdfs"):
    pdf_files = glob.glob(f"{pdf_folder}/*.pdf")  # Get all PDF files in the directory
    combined_text = get_text_from_pdfs(pdf_files)
    text_chunks = get_text_chunks(combined_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a virtual assistant for Vivekananda Institute of Professional Studies - Technical Campus (VIPS-TC).
    Answer the question as detailed as possible from the provided context. If the answer is not in
    provided context just say, "I'm sorry, but I don't have that specific information about VIPS-TC. 
    You may want to check the official VIPS website or contact the institute directly for the most up-to-date and accurate information."
    Don't provide any incorrect information.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process the user's question and retrieve the answer
def process_question(user_question):
    # Preprocess question to handle common variations
    user_question = preprocess_question(user_question)
    
    # Check if the question is about a timetable
    timetable_keywords = ["timetable", "schedule", "time table"]
    for branch, path in timetable_paths.items():
        if any(keyword in user_question.lower() for keyword in timetable_keywords) and branch in user_question.lower():
            return path  # Return the path to the specific timetable image

    # Handle basic conversational questions
    if re.search(r"\bhello\b|\bhi\b|\bhow are you\b|\bwhat's up\b|\bhey\b", user_question, re.IGNORECASE):
        return "Hello! How can I assist you with VIPS-TC-related queries today?"

    try:
        # Process document-based questions
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Use keywords to prioritize syllabus-related searches
        keywords = ["syllabus", "subject", "course"]
        if any(keyword in user_question.lower() for keyword in keywords):
            relevant_docs = new_db.similarity_search(user_question, k=5)  # Increase 'k' if needed
        else:
            relevant_docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    
    except Exception as e:
        # Return a friendly error message if something goes wrong
        return f"I'm sorry, but I encountered an issue retrieving the information: {str(e)}"

# Streamlit app for the chatbot
def main():
    st.set_page_config(page_title="VIPS Virtual Assistant", page_icon="🤖")
    
    st.title("VIPS Virtual Assistant 🤖")
    st.write("Hey there! I'm your virtual assistant for VIPS. How can I help you today?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me anything about VIPS-TC"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_question(prompt)
                if response.endswith(".png") or response.endswith(".jpg"):  # Check if response is an image path
                    st.image(response)  # Display image if path is returned
                else:
                    st.markdown(response)  # Display text response otherwise
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Load and process the PDF files (only once)
    if "pdf_processed" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            initialize_vector_store()  # Load all PDFs from the specified folder
            st.session_state.pdf_processed = True

if __name__ == "__main__":
    main()
