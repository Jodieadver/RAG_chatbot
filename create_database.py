# turning internal materials into vector DB for retrieval



from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


import os
import shutil



DATA_PATH = 'data'
CHROMA_PATH = "chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob='*.md')
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # 安全打印一个样例
    if len(chunks) > 0:
        idx = min(10, len(chunks) - 1)
        document = chunks[idx]
        print(document.page_content[:500])  # 只打印前500字符，避免太长
        print(document.metadata)
    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding = HuggingFaceEmbeddings(
        model_name= MODEL_NAME
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,          
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Directory not found: '{DATA_PATH}'")
    documents = load_documents()
    if not documents:
        print("No documents found. Please put .md files under the 'data' folder.")
        return
    chunks = split_text(documents)
    if chunks:
        save_to_chroma(chunks)
    else:
        print("No chunks to save.")


def main():
    generate_data_store()


if __name__ == "__main__":
    main()
    
