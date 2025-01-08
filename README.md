# 📄 Chat & Talk with PDF Using Streamlit 🌟

Welcome to the **Chat & Talk with PDF** project! This application allows you to interact with one or multiple PDFs using both **text-based chat** and **voice inputs**. Built with **LangChain framework**, it leverages the power of **Google Gemini-pro** for conversational AI and **OpenAI Whisper (via Groq)** for voice transcription. 🎙️📚✨

---

## 🎯 **Features**
1. **Chat with PDFs**: Ask questions and get accurate answers from the content of uploaded PDFs.  
2. **Multiple PDFs Supported**: Process and interact with multiple PDFs at once!  
3. **Voice Interaction**: Record your voice, transcribe it, and use the transcribed text to interact with the PDFs.  
4. **Efficient Search**: Uses a vector database to search and retrieve answers efficiently.  
5. **Streamlit UI**: A sleek, simple, and user-friendly interface.  

---

## 🛠️ **Technologies Used**
- **Frontend**: Streamlit 🌟  
- **PDF Processing**: PyPDF2 📄  
- **Voice Transcription**: OpenAI Whisper (via Groq) 🎙️  
- **Vector Database**: FAISS 📊  
- **Conversational AI**: LangChain and Google Gemini-pro 🤖  
- **Environment Variables**: dotenv 🔑  
- **Audio Handling**: SoundDevice and Wave 📼  

---

## 🚀 **How It Works**
1. **Upload PDFs**: Drag and drop one or more PDF files into the sidebar.  
2. **Process PDFs**: The application extracts text from the PDFs, splits it into chunks, and saves it in a searchable vector database.  
3. **Ask Questions**:  
   - **Text Input**: Type your question into the chat interface to get answers.  
   - **Voice Input**: Record your voice, let the app transcribe it, and generate an answer based on the transcribed text.  
4. **Get Answers**: Answers are generated using AI models trained to retrieve and summarize relevant information from the context.  

---

## ⚙️ **Setup and Installation**
Follow these steps to get the app running on your local system:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   - Create a `.env` file in the root directory.  
   - Add your **Google API Key** and **Groq API Key**:  
     ```plaintext
     GOOGLE_API_KEY=your-google-api-key
     GROQ_API_KEY=your-groq-api-key
     ```

4. **Run the App**:
   Launch the Streamlit app with the following command:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open the link displayed in the terminal, usually `http://localhost:8501`, and start chatting with your PDFs! 🎉  

---

## 🎥 **Demo**


---

Happy Chatting & Talking with PDFs! 🎉📖🗣️