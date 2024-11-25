import os
import json
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from src.googledrive_download_folder import *
from pinecone import Pinecone
from uuid import uuid4


def load_env_variables(env_file_path):
    """Load environment variables from the specified file."""
    load_dotenv(env_file_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = openai_api_key

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

    return openai_api_key, pinecone_api_key


def download_files_from_google_drive(google_drive_json, destination_folder="tmp"):
    """Download files from Google Drive based on the provided JSON."""
    with open(google_drive_json, 'r') as f:
        googledrive_folders = json.load(f)

    for company, folder_id in googledrive_folders.items():
        print(f"Downloading files for {company}...")
        download_and_process_folder(folder_id, f"{destination_folder}/{company}")

def collect_downloaded_html_files_path(root_folder):
    """Collect all HTML files recursively from the specified root folder."""
    html_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))

    return html_files

def load_html_documents(file_paths):
    """Load documents from a list of HTML file paths."""
    docs = []
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        loader = UnstructuredHTMLLoader(file_path)
        docs.extend(loader.load())
    return docs

# def build_vectorstore(docs, persist_directory, model="text-embedding-3-small"):
#     """Build or load a vectorstore from documents."""

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
#     all_splits = text_splitter.split_documents(docs)

#     # Save and load vectorstore
#     Chroma.from_documents(documents=all_splits, embedding = OpenAIEmbeddings(model=model), persist_directory=persist_directory)
    
#     # Delete /tmp folder
#     shutil.rmtree("tmp")
    
#     return 

# def get_vectorstore(persist_directory, model="text-embedding-3-small"):
#     """Load a vectorstore from the specified directory."""

#     return Chroma(persist_directory=persist_directory, embedding_function = OpenAIEmbeddings(model=model))

def build_vectorstore(docs, index_name, pinecone_api_key, model="text-embedding-3-small"):
    """Build a Pinecone vectorstore from documents."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)

    embedding_model = OpenAIEmbeddings(model=model)
    
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    uuids = [str(uuid4()) for _ in range(len(all_splits))]
    vector_store.add_documents(documents=all_splits, ids=uuids)

    # Delete /tmp folder
    shutil.rmtree("tmp")

    return vector_store

def get_vectorstore(index_name, pinecone_api_key, model="text-embedding-3-small"):
    """Load a Pinecone vectorstore from the specified index name."""

    embedding_model = OpenAIEmbeddings(model=model)
    
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    return vector_store


def create_rag_chain(vectorstore, model="gpt-4o-mini"):
    """Create a RAG (Retrieval-Augmented Generation) chain."""

    # Define LLM
    llm = ChatOpenAI(model=model)

    # Define Retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Contextualized prompt for rephrasing the question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA chain for generating answers
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Keep the answer concise.\n\n{context}"
            )),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_answer_and_update_history(rag_chain, question, chat_history, max_history = 8):
    """Retrieve an answer for a question using the RAG chain."""

    response = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=response["answer"]),
    ])

    # Truncate chat history to the last 'max_history' messages
    if len(chat_history) > max_history:
        chat_history[:] = chat_history[-max_history:]  # Keep only the last max_history messages

    return response["answer"]

