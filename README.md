# CampusMate-VIPS Virtual Assistant 🤖  

A chatbot built for **VIPS-TC** using **Streamlit, Google Generative AI, and FAISS**. The assistant helps students with queries related to **syllabus, timetable, faculty details, course information, and general college inquiries** by processing documents and user input intelligently.  

## 🚀 Features  
- **Natural Language Processing** for answering student queries.  
- **Timetable Retrieval** based on branch-specific requests.  
- **PDF Document Processing** to extract relevant syllabus and course details.  
- **FAISS Vector Store** for efficient semantic search.  
- **Google Generative AI (Gemini)** for conversational responses.  
- **Streamlit Web Interface** for an interactive user experience.  

## 🛠️ Tech Stack  
- **Frontend**: Streamlit  
- **Backend**: Python  
- **AI Models**: Google Generative AI (Gemini)  
- **Vector Storage**: FAISS  
- **PDF Processing**: PyPDF2  
- **Embeddings**: GoogleGenerativeAIEmbeddings

## 📌 How It Works
- The assistant loads and processes college-related PDFs (syllabus, rules, etc.).
- It uses FAISS for semantic search across documents.
- Users can ask queries via the Streamlit chat interface.
- It intelligently retrieves relevant information or provides a response using Gemini AI.
- Users can also fetch timetables by mentioning their branch.
