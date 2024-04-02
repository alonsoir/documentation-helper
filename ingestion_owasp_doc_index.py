import os

from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredImageLoader,
    UnstructuredFileLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

INDEX_NAME = "owasp-cheatsheets-doc-index"


def ingest_docs():
    # Load HTML files from the 'site-owasp-cheatsheet/cheatsheets' directory
    html_loader = ReadTheDocsLoader("site-owasp-cheatsheet/cheatsheets")
    html_documents = html_loader.load()

    # Load PDF files from the 'site-owasp-cheatsheets/assets' directory
    # pdf_loader = UnstructuredPDFLoader("site-owasp-cheatsheets/assets")
    # pdf_documents = pdf_loader.load()

    # Load PNG files from the 'site-owasp-cheatsheets/assets' directory
    # image_loader = UnstructuredImageLoader("site-owasp-cheatsheets/assets")
    # image_documents = image_loader.load()

    # Load Java files from the 'site-owasp-cheatsheets/assets' directory
    #java_loader = UnstructuredFileLoader(
    #    "site-owasp-cheatsheets/assets", blob="*.java"
    #)
    # java_documents = java_loader.load()

    # Combine all the documents
    # raw_documents = html_documents + pdf_documents + image_documents + java_documents
    raw_documents = html_documents
    print(f"loaded {len(raw_documents)} documents")

    # Split the documents into smaller chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    # documents = text_splitter.split_documents(raw_documents)
    for doc in raw_documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("site-owasp-cheatsheet", "https:/")
        doc.metadata.update({"source": new_url})

    # Create embeddings and add the documents to the Pinecone index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Going to add {len(raw_documents)} to Pinecone")
    PineconeLangChain.from_documents(raw_documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
