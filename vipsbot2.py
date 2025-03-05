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
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import threading
import queue

# Load environment variables (Google API key)
load_dotenv()

# Configure the Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Global variables
WAKE_WORD = "hey vips"
audio_queue = queue.Queue()
is_listening = False
assistant_state = "Idle"

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

# Function to convert text to speech and play it
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio = AudioSegment.from_mp3(fp.name)
        play(audio)
        os.unlink(fp.name)

# Function to listen for the wake word
def listen_for_wake_word(recognizer, microphone):
    global is_listening, assistant_state
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            if WAKE_WORD in text:
                is_listening = True
                assistant_state = "Listening"
                play(AudioSegment.from_wav("wake_sound.wav"))  # Play a sound to indicate activation
                return
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")

# Function to capture voice input after wake word detection
def capture_voice_input(recognizer, microphone):
    global is_listening, assistant_state
    assistant_state = "Listening for question"
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, phrase_time_limit=10)
    try:
        text = recognizer.recognize_google(audio)
        is_listening = False
        assistant_state = "Processing"
        return text
    except sr.UnknownValueError:
        is_listening = False
        assistant_state = "Idle"
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        is_listening = False
        assistant_state = "Idle"
        return "Sorry, there was an error processing your speech."

# Function to handle voice interaction
def voice_interaction_thread():
    global assistant_state
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    while True:
        assistant_state = "Waiting for wake word"
        listen_for_wake_word(recognizer, microphone)
        if is_listening:
            user_input = capture_voice_input(recognizer, microphone)
            if user_input != "Sorry, I couldn't understand that." and user_input != "Sorry, there was an error processing your speech.":
                audio_queue.put(user_input)

# Function to initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    pdf_doc = 'VIPS_QA.pdf'
    raw_pdf_text = get_pdf_text(pdf_doc)
    text_chunks_pdf = get_text_chunks(raw_pdf_text)
    get_vector_store(text_chunks_pdf)
    return True

# Streamlit app for the chatbot
def main():
    st.set_page_config(page_title="VIPS Virtual Assistant", page_icon="🤖")
    
    st.title("VIPS Virtual Assistant 🤖")
    st.write("Say 'Hey VIPS' to activate the voice assistant!")

    # Initialize chat history and state indicator
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    state_indicator = st.empty()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize the chatbot
    with st.spinner("Initializing chatbot..."):
        initialize_chatbot()

    # Start voice interaction thread
    voice_thread = threading.Thread(target=voice_interaction_thread, daemon=True)
    voice_thread.start()

    while True:
        # Update state indicator
        state_indicator.text(f"Assistant State: {assistant_state}")

        # Process voice input from queue
        if not audio_queue.empty():
            user_input = audio_queue.get()
            st.write(f"You said: {user_input}")
            
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

        # React to text input (keep this for typing option)
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

        # Rerun the app to update the UI
        st.rerun()

if __name__ == "__main__":
    main()