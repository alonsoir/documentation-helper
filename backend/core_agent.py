import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec

load_dotenv()  # take environment variables from .env.

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
print(f"PINECONE_API_KEY is {PINECONE_API_KEY}")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "langchain-doc-index"


def run_llm_with_agent(query: str, chat_history: List[Dict[str, Any]] = []):
    openai_api_key = os.environ.get("OPENAI_API_KEY") or "OPENAI_API_KEY"
    model_name = "text-embedding-3-small"
    pinecone_index_name = "langchain-doc-index"
    # initialize connection to pinecone (get API key at app.pc.io)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY") or "PINECONE_API_KEY"
    environment = (
            os.environ.get("PINECONE_ENVIRONMENT_REGION") or "PINECONE_ENVIRONMENT_REGION"
    )
    # configure client
    pc = Pinecone(api_key=pinecone_api_key)
    use_serverless = os.environ.get("USE_SERVERLESS", "False").lower() == "true"

    if use_serverless:
        spec = ServerlessSpec(cloud="aws", region="us-west-2")
    else:
        spec = PodSpec(environment=environment)

    index = pc.Index(pinecone_index_name)
    print(index.describe_index_stats())

    if pinecone_index_name in pc.list_indexes().names():
        print(pc.list_indexes())

    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

    text_field = "text"

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index_name, embedding=embeddings, text_key=text_field
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})
