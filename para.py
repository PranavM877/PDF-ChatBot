import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Query With PDF", page_icon="📄", layout="wide")

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store using embeddings from Google Generative AI
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load QA conversational chain for answering questions
def get_conversational_chain():
    prompt_template1 = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context." Do not provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template1, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and provide response using the vector store and conversational chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Response:\n\n", response["output_text"])

# Get response from Gemini model
def get_gemini_response(input_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_text)
    return response.text

# Main function for Streamlit app
def main():
    # Apply indigo-black theme using custom CSS and hide the top orange line
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
        }
        .stApp {
            background-color: black;
            color: white;
        }
        .css-18e3th9 {
            background-color: indigo;
        }
        .stButton>button {
            background-color: indigo;
            color: white;
        }
        .stButton>button:hover {
            background-color: #4b0082;
            color: white;
        }
        header {
            visibility: hidden;
        }
        .equal-height {
            display: flex;
            flex-direction: column;
            height: 100%; /* Ensure full height */
        }
        .container-style {
            background-color: #2e2e2e; /* Grey background */
            padding: 20px;
            flex-grow: 1; /* Make the container grow to fill space */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Query With Your Uploaded PDFs")

    # Create two side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        # Container for PDF upload section
        with st.container():
            st.markdown('<div class="equal-height">', unsafe_allow_html=True)
            st.header("Upload Your PDFs")
            pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
            
            if st.button("Submit & Process", key="submit"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")

            if st.button("Get Summary", type="primary", key="keypoints"):
                input_prompt = """fetch all key points in detail from the given text information. Do not give extra information; only give information relevant to the text."""
                raw_text = get_pdf_text(pdf_docs)
                response = get_gemini_response(raw_text + " as text information " + input_prompt)
                st.subheader(response)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Get 3 Questions", type="primary", key="3questions"):
                input_prompt = """Generate 3 questions related to the uploaded document, begin your answer with Questions: and then provide the questions from the next line"""
                raw_text = get_pdf_text(pdf_docs)
                response = get_gemini_response(raw_text + " as text information " + input_prompt)
                st.subheader(response)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Container for query input section
        with st.container():
            st.markdown('<div class="equal-height">', unsafe_allow_html=True)
            st.header("Ask a Question")
            user_question = st.text_input("Ask a Question from the PDF Files")

            if user_question:
                user_input(user_question)
            st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()