from log_config import get_logger
logger = get_logger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pandas as pd
import os

# Corrigido: agora aponta direto para a raiz 'app/'
BASE_DIR = "app/"
ARQUIVO_EXCEL = os.path.join(BASE_DIR, "parcerias.xlsx")
INDEX_DIR = os.path.join(BASE_DIR, "vectorstore_canais/")

# Garante apenas o diretório do índice
os.makedirs(INDEX_DIR, exist_ok=True)

def carregar_documentos():
    """Carrega todas as abas da planilha Excel como documentos."""
    if not os.path.exists(ARQUIVO_EXCEL):
        raise FileNotFoundError(f"Arquivo Excel não encontrado em: {ARQUIVO_EXCEL}")

    logger.info("Lendo planilha Excel com abas de conhecimento...")
    df_dict = pd.read_excel(ARQUIVO_EXCEL, sheet_name=None)
    documentos = []

    for aba, df in df_dict.items():
        for _, row in df.iterrows():
            linha = " ".join(str(val) for val in row if pd.notna(val))
            if linha.strip():
                documentos.append({"content": linha, "source": aba})

    logger.info(f"Total de linhas lidas: {len(documentos)}")
    return documentos

def indexar_documentos():
    """Cria o índice vetorial FAISS com embeddings da OpenAI a partir do Excel."""
    try:
        dados = carregar_documentos()
        if not dados:
            raise ValueError("Nenhum conteúdo válido foi encontrado no Excel.")

        docs = [d["content"] for d in dados]
        metadados = ["Origem: " + d["source"] for d in dados]

        from langchain.schema import Document
        documentos = [Document(page_content=doc, metadata={"source": meta}) for doc, meta in zip(docs, metadados)]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documentos)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)

        logger.info(f"Indexação concluída com {len(documentos)} documentos e {len(chunks)} chunks.")
        return len(documentos)
    except Exception as e:
        logger.error(f"Erro ao indexar documentos: {e}")
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
    print(f"✅ {qtd} registros indexados com sucesso.")
    resultado = buscar_conhecimento("Quais os modelos ideais de canais?")
    print("\n📌 RESPOSTA:\n", resultado)
