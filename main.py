from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.schema import Document
import requests
import io
import os



# FastAPI app setup
app = FastAPI()

# Input model for query
class QueryRequest(BaseModel):
    query: str

# Google API setup
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

def create_db_from_github_file(file_url: str, file_type: str = "text") -> FAISS:
    """
    Create a FAISS database from a file hosted on GitHub.

    Args:
        file_url (str): URL of the file on GitHub.
        file_type (str): Type of the file ('text' or 'pdf'). Default is 'text'.

    Returns:
        FAISS: A FAISS vector store database built from the file's content.
    """
    # Fetch the file from GitHub
    response = requests.get(file_url)
    response.raise_for_status()  # Raise an error if the request fails
    file_content = response.content

    # Load the document based on file type
    if file_type == "text":
        # Directly create a document for plain text
        text = file_content.decode("utf-8")
        documents = [Document(page_content=text, metadata={"source": file_url})]
    elif file_type == "pdf":
        # Use PyPDFLoader directly from a bytes stream for PDF
        loader = PyPDFLoader(io.BytesIO(file_content))
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create the FAISS vector store
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    return retriever

# Load the retriever (initialize it only once)
retriever = create_db_from_github_file("https://raw.githubusercontent.com/abdullah0150/Images/refs/heads/main/data", file_type="text")

# Function to generate response based on query
def get_response_from_query(retriever, query):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""Answer the following question: {question}
        Based on these info: {context}

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose, detailed.
        """,
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)

# FastAPI endpoint for answering queries
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        query = request.query
        response = get_response_from_query(retriever, query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))