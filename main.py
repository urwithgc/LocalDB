import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def initialize_groq_llm():
    # Initialize the Groq language model with API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key is missing. Please check your .env file.")
        return None
    return Groq(api_key=api_key)

def load_vector_store(pdf_path, embeddings, store_name):
    # Load existing vector store or create a new one from the PDF
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        # Extract text from PDF and create vector store
        pdf_reader = PdfReader(pdf_path)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        return vector_store

def main():
    # Main function to run the Streamlit app
    st.title("Simple RAG Application")

    llm = initialize_groq_llm()
    if llm is None:
        return  # Exit if the API key is not available

    pdf_path = "document.pdf"

    # Define embedding models
    embeddings_models = {
        '300-dim': HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        '700-dim': HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"),
        '1536-dim': HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    }

    # Load vector stores for each embedding model
    vector_stores = {}
    for name, embeddings in embeddings_models.items():
        vector_stores[name] = load_vector_store(pdf_path, embeddings, f"vector_store_{name}")

    query = st.text_input("**Ask your question:**")

    if query:
        responses = {}
        # Search and get responses from each vector store
        for name, vector_store in vector_stores.items():
            st.write(f"Searching in vector store: {name}")
            docs = vector_store.similarity_search(query=query, k=3)

            if not docs:
                st.write(f"No documents found in {name} vector store.")
                responses[name] = "No documents found."
                continue

            st.write(f"Found {len(docs)} documents in {name} vector store.")
            snippets = " ".join([doc.page_content for doc in docs])
            st.write(f"Document snippets: {snippets}")

            # Generate a response from the Groq language model
            prompt = f"Given the following document snippets, provide a detailed and relevant response to the query: '{query}'.\n\nDocument Snippets:\n{snippets}"
            try:
                result = llm.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "Provide detailed and clear responses based on the provided document snippets."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_content = result.choices[0].message.content
            except Exception as e:
                st.write(f"Error occurred: {e}")
                response_content = "An error occurred while generating the response."

            responses[name] = response_content

        # Display responses from different embedding models
        st.subheader("Responses from Different Embeddings:")
        for name, response in responses.items():
            st.write(f"**{name}**:")
            st.write(response)

if __name__ == '__main__':
    main()
