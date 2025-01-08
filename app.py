##Import packages
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile
import sounddevice as sd
import wave

load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Groq and Whisper Setup
api_key_groq = os.getenv("GROQ_API_KEY")
if not api_key_groq:
    raise ValueError("API key for Groq is not set in the environment variables.")

client = Groq(api_key=api_key_groq)

# Convert audio file to text using Whisper
def get_whisper(audio_file_path):
    """Transcribe audio into text using Whisper."""
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=audio_file,
            language="en"
        )
    return response.text if response.text else "Error processing the audio"

# Extract all text from uploaded PDF files
def extract_text_from_pdf(pdf_docs):
    """Read and extract text from all uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Break large text into smaller chunks for processing
def get_text_chunks(text):
    """Split long text into smaller, manageable pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Save text chunks into a searchable database
def store_in_vectordb(text_chunks):
    """Store text chunks in a searchable vector database."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Set up the chat model with a custom prompt
def fetch_conversational_chain():
    """Create a chatbot that answers questions based on provided context."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context."
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Search the database for relevant information and get an answer
def user_input(user_question):
    """Search the database and answer user questions."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = fetch_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Record audio from the microphone and save it as a file
def record_voice(file_path):
    """Record voice using a microphone and save it as a .wav file."""
    st.write("Recording... Speak now!")
    duration = 5  # Duration in seconds
    fs = 16000  # Sample rate
    recorded_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recorded_data.tobytes())

# Main Streamlit App
def main():
    """Streamlit app for chatting with PDF content and using voice transcription."""
    st.set_page_config("Chat PDF & Voice")
    st.header("Chat and talk with your PDF! Featuring Gemini🌞")

    # Sidebar Menu
    with st.sidebar:
        st.title("Processing Area📇")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button 🌏", accept_multiple_files=True)
        if st.button("Submit 🛎️"):
            with st.spinner("Baking🔥🔥🔥"):
                raw_text = extract_text_from_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                store_in_vectordb(text_chunks)
                st.success("PDF processing done!😃")

    # Voice Transcription Section
    st.subheader("Talk 🙋🏻‍♂️")
    if st.button("Start Recording..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            record_voice(temp_audio.name)
            transcription = get_whisper(temp_audio.name)
            user_input(transcription)
            st.success("Transcription Complete!")
            st.write("Transcribed Text: ", transcription)

    # Chat Section
    st.subheader("Chat with PDF💖")
    user_question = st.text_input("I am the pdf. Ask me!🤔")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
