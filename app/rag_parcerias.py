from log_config import get_logger
logger = get_logger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pathlib import Path
import os

BASE_DIR = "app/conhecimento_canais/"
INDEX_DIR = "app/vectorstore_canais/"

# Garante que os diretórios existem
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def carregar_documentos():
    """Carrega todos os arquivos .txt da base de conhecimento."""
    arquivos = list(Path(BASE_DIR).glob("*.txt"))
    docs = []
    for arq in arquivos:
        loader = TextLoader(str(arq), encoding="utf-8")
        docs.extend(loader.load())
    return docs

def indexar_documentos():
    """Cria o índice vetorial FAISS com embeddings da OpenAI."""
    try:
        documentos = carregar_documentos()
        if not documentos:
            raise ValueError("Nenhum documento .txt foi encontrado em 'app/conhecimento_canais/'.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documentos)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)

        return len(documentos)
    except Exception as e:
        raise RuntimeError(f"Erro ao indexar documentos: {e}")

def carregar_index():
    """Carrega o índice FAISS salvo."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def buscar_conhecimento(pergunta: str, k=3) -> str:
    """Realiza uma busca semântica na base indexada e retorna os melhores trechos."""
    db = carregar_index()
    docs = db.similarity_search(pergunta, k=k)
    resposta = "\n\n".join(doc.page_content.strip() for doc in docs)
    return resposta

# Teste local
if __name__ == "__main__":
    qtd = indexar_documentos()
    print(f"✅ {qtd} arquivos indexados com sucesso.")
    resultado = buscar_conhecimento("Qual o melhor modelo de canal para produto complexo?")
    print("\n📌 RESPOSTA:\n", resultado)
