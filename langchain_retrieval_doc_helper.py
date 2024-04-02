import os
import time
from typing import List

from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec

load_dotenv()  # take environment variables from .env.

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

pinecone_index_name = "langchain-doc-index"

index = pc.Index(pinecone_index_name)
print(index.describe_index_stats())

if pinecone_index_name in pc.list_indexes().names():
    print(pc.list_indexes())

# we i have to create a new index?
# pc.create_index(
#        index_name,
#        dimension=1536,  # dimensionality of text-embedding-ada-002
#        metric='dotproduct',
#        spec=spec
#    )

# in my case, no, because i already created using the web api and initialized using ingestion_langchain_doc_index.py script.

# wait for index to be initialized
while not pc.describe_index(pinecone_index_name).status["ready"]:
    time.sleep(1)


openai_api_key = os.environ.get("OPENAI_API_KEY") or "OPENAI_API_KEY"
model_name = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

text_field = "text"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=pinecone_index_name, embedding=embeddings, text_key=text_field
)

my_query = "types of agents that Langchain can use"

someSimilarDocs: List[Document] = docsearch.similarity_search(query=my_query, k=5)
print(f"someSimilarDocs is: {someSimilarDocs}")

anotherQuery = "types of agents that Langchain can use"
docs: List[Document] = docsearch.similarity_search(anotherQuery)

print(f"docs is: {docs}")

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

qa.run(my_query)

tools = [
    Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about the topic"
        ),
    )
]


agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversational_memory,
)

agent(anotherQuery)

agent("what is 2 * 7?")

agent("can you tell me about the different types of agents that langchain can use?")

agent("Puedes decirme el proposito del uso de langchain?")

# pc.delete_index(index_name)

print("Done!")
