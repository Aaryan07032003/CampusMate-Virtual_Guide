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
from gtts import gTTS
import tempfile
import os

# Load environment variables (Google API key)
load_dotenv()

# Configure the Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Try to import speech_recognition and playsound
try:
    import speech_recognition as sr
    from playsound import playsound
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False

# Function to extract text from PDF file
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save FAISS index locally

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
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process the user's question and retrieve the answer
def process_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS index with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# Function to capture voice input (only if speech_recognition is available)
def get_voice_input():
    if not VOICE_INPUT_AVAILABLE:
        return "Voice input is not available. Please install the required libraries."
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("Listening... Speak now."):
            audio = r.listen(source)
        with st.spinner("Processing your speech..."):
            try:
                text = r.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand that."
            except sr.RequestError:
                return "Sorry, there was an error processing your speech."

# Updated function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        try:
            if VOICE_INPUT_AVAILABLE:
                playsound(fp.name)
            else:
                st.audio(fp.name, format='audio/mp3')
        except Exception as e:
            st.warning(f"Unable to play audio due to an error: {str(e)}")
            st.write("Fallback: Playing audio using Streamlit's built-in player")
            st.audio(fp.name, format='audio/mp3')
        finally:
            os.unlink(fp.name)

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

    # Add voice input button with microphone widget if available
    if VOICE_INPUT_AVAILABLE:
        col1, col2 = st.columns([1, 4])
        with col1:
            speak_button = st.button("🎤")
        with col2:
            st.write("Click the microphone to speak your question")
        
        if speak_button:
            user_input = get_voice_input()
            st.write(f"You said: {user_input}")
            
            if user_input != "Sorry, I couldn't understand that." and user_input != "Sorry, there was an error processing your speech.":
                # Process the question and get the response
                with st.spinner("Thinking..."):
                    response = process_question(user_input)
                
                # Display user message and response
                st.chat_message("user").markdown(user_input)
                st.chat_message("assistant").markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Convert response to speech
                text_to_speech(response)
    else:
        st.write("Voice input is not available. Please use text input.")

    # React to text input
    if prompt := st.chat_input("Type your question here"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_question(prompt)
            st.markdown(response)
            # Convert response to speech
            text_to_speech(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Load and process the PDF file (do this only once)
    if "pdf_processed" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            pdf_doc = 'VIPS_QA.pdf'
            raw_pdf_text = get_pdf_text(pdf_doc)
            text_chunks_pdf = get_text_chunks(raw_pdf_text)
            get_vector_store(text_chunks_pdf)
            st.session_state.pdf_processed = True

if __name__ == "__main__":
    main()