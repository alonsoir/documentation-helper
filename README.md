
# LangChain Documentation Helper

A repository for learning LangChain by building a generative ai application. Original work from Eden Marco.

This is a web application demo that use Pinecone as a vectorstore and answers questions about LangChain.
Pinecone act a context database, first you have to download the docs from sources from LangChain official documentation. 
Obviously you can do the same with every dataset you want.  

Basically you have three scripts in this project:

langchain_retrieval_doc_helper.py --> my work. Original work from Eden was not working for me.
main.py --> it invokes the logic from core_agent.py
core_agent.py --> basically is a refactor from first script.

![Logo](https://github.com/emarco177/documentation-helper/blob/main/static/banner.gif)
[![udemy](https://img.shields.io/badge/LangChain%20Udemy%20Course-%2412.99-green)](https://www.udemy.com/course/langchain/?couponCode=LANGCHAINCD8C0B4060)

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`PINECONE_API_KEY`
`PINECONE_ENVIRONMENT_REGION`
`OPENAI_API_KEY`

## Run Locally

Clone the project

```bash
  git clone https://github.com/alonsoir/documentation-helper.git
```

Go to the project directory

```bash
  cd documentation-helper
```

Download LangChain Documentation
```bash
  mkdir langchain-docs
  wget -r -A.html -P langchain-docs  https://api.python.langchain.com/en/latest
```
Run ingestion script to create the index in Pinecone. https://app.pinecone.io
My index is langchain-doc-index, be sure to match this name with the one you use in the code. 
```
    python ingestion.py
```
Install dependencies

```bash
  pipenv install
```

Start the flask server

```bash
  streamlit run main.py
```

## Running Tests

To run tests, run the following command

```bash
  pipenv run pytest .
```


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.udemy.com/course/langchain/?referralCode=D981B8213164A3EA91AC)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eden-marco/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://www.udemy.com/user/eden-marco/)
