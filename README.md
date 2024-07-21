# Chat with PDF - README

## Overview

This Streamlit application allows users to upload PDF documents and interact with the content through a conversational interface. It uses Google Generative AI, LangChain, and FAISS to deliver context-aware responses based on user queries.

## Requirements

Ensure you have the following installed before running the application:

- Python 3.8 or later
- Streamlit
- PyPDF2
- dotenv
- LangChain
- FAISS
- langchain_google_genai
- langchain_community


You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the Repository**

   Clone the repository containing the `chat with pdf.py` file to your local machine.

   ```bash
   git clone https://github.com/hritikpal671/Chat-with-PDF.git
   cd <repository_directory path>
   ```

2. **Set Up Environment Variables**

   Create a `.env` file in the root directory of the project and add your Google API key:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

3. **Running the Application**

   Execute the following command to start the Streamlit application:

   ```bash
   streamlit run "chat with pdf.py"
   ```

4. **Using the Application**

   - Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).
   - Upload a PDF document using the file uploader.
   - Interact with the document by typing your questions into the chat input.

## Code Explanation

- **Environment Configuration**: The application loads environment variables using `dotenv` to securely manage the API key.
- **Google Generative AI**: Configures the Generative AI model (Gemini 1.5 Flash) to generate responses.
- **LangChain Integration**: Utilizes LangChain to manage conversation history and context.
- **FAISS Vector Store**: Embeds the PDF content and stores it in FAISS for efficient retrieval.
- **Streamlit Interface**: Provides a simple interface for uploading PDFs and chatting with the document content.

## Troubleshooting

- **Environment Variables**: Ensure your `.env` file is correctly set up with a valid Google API key.
- **Dependencies**: Verify that all required dependencies are installed.
- **Network Issues**: Make sure your internet connection is stable, as the application relies on Google Generative AI.
