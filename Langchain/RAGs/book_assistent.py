import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"Documents","MyBook.txt")
persistent_directory = os.path.join(current_dir,"DB","chroma_db")


if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("\nSplitting documents into chunks...")
    print(f"Number of chunks created: {len(docs)}")
    print(f"Sample chunk:\n {docs[0].page_content}")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    print("finished creating embeddings.")

    print("Creating vector store...")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("Vector store created successfully.")

else:
    print("vector Store already exists.")

