from log_config import get_logger
logger = get_logger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from pathlib import Path
import os
import pandas as pd

BASE_DIR = "app/conhecimento_canais/"
INDEX_DIR = "app/vectorstore_canais/"
ARQUIVO_EXCEL = os.path.join(BASE_DIR, "conhecimento_canais_completo.xlsx")

# Garante que os diretórios existem
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def carregar_documentos():
    """Carrega os dados da planilha Excel como documentos do LangChain."""
    if not os.path.exists(ARQUIVO_EXCEL):
        raise FileNotFoundError(f"Arquivo Excel não encontrado em: {ARQUIVO_EXCEL}")

    df_dict = pd.read_excel(ARQUIVO_EXCEL, sheet_name=None)  # Lê todas as abas
    documentos = []

    for aba, df in df_dict.items():
        if df.empty or df.shape[1] == 0:
            continue
        conteudo = "\n".join(str(val) for val in df.iloc[:, 0].dropna())  # Assume coluna A
        documentos.append(Document(page_content=conteudo, metadata={"fonte": aba}))

    return documentos

def indexar_documentos():
    """Cria o índice vetorial FAISS com embeddings da OpenAI a partir da planilha."""
    try:
        documentos = carregar_documentos()
        if not documentos:
            raise ValueError("Nenhum conteúdo foi carregado da planilha Excel.")

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
    print(f"✅ {qtd} abas da planilha foram indexadas com sucesso.")
    resultado = buscar_conhecimento("Qual o melhor modelo de canal para produto complexo?")
    print("\n📌 RESPOSTA:\n", resultado)

