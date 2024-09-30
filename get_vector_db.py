import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')

def get_vector_db():
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL,show_progress=True)
    
    """
    embedding = HuggingFaceEndpointEmbeddings(
    model= "mixedbread-ai/mxbai-embed-large-v1",
    task="feature-extraction",
    huggingfacehub_api_token="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")
    """

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    return db