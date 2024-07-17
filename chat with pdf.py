from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
import warnings
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
warnings.filterwarnings("ignore")
from langchain_google_genai import GoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import PyPDF2

load_dotenv()  # loading all the environment variables

# Configure Google Generative AI
api = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])
model1 = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api)

# generating response using Google's Gemini 1.5 Flash model
def get_gemini_response(question, chat_history):
    context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-10:]])
    question_with_context = f"{context}\nUser: {question}"
    response = chat.send_message(question_with_context, stream=True)
    return response

# extracting text from pdf
def parse_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Initialize our Streamlit app
st.set_page_config(page_title="Chat with Document")

st.header("Chat with your PDF")

# Set a default model
if "google_gemini" not in st.session_state:
    st.session_state["google_gemini"] = "gemini-1.5-flash"

# Add a Gemini Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Initialize session state for chat history and document text if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'document_text' not in st.session_state:
    st.session_state['document_text'] = ""

if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None

# Initialize chat history and context memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize conversation summary memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(llm=model1)

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

# Inside the file upload section
if uploaded_file:
    st.session_state['document_text'] = parse_pdf(uploaded_file)
    document = Document(page_content=st.session_state['document_text'])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api)
    vectorstore = FAISS.from_documents([document], embeddings)
    st.session_state['qa_chain'] = ConversationChain(
        llm=model1,
        memory=st.session_state.memory
    )
    st.write("Document content loaded. You can now ask questions based on this document.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
input_text = st.chat_input("Hi, please ask your question")

if input_text:
    # Display user message
    with st.chat_message("user"):
        st.markdown(input_text)
    
    if st.session_state['qa_chain']:
        # Pass input_text to the chain and get the result
        result = st.session_state['qa_chain'].predict(input=input_text)
        response_text = result  # Use the result directly
    else:
        response = get_gemini_response(input_text, st.session_state.messages)
        response_text = "".join([chunk.text for chunk in response])

    # Add user query and response to session state chat history
    st.session_state.messages.append({"role": "user", "content": input_text})
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_text)
