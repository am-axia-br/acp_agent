from log_config import get_logger
logger = get_logger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.text_splitter import NLTKTextSplitter

from pathlib import Path
import pandas as pd
import os
import time
import threading

from segmento_equivalencias import *

# Configurações globais
ARQUIVO_EXCEL = "canais.xlsx"
INDEX_DIR = "vectorstore_canais/"
os.makedirs(INDEX_DIR, exist_ok=True)

# --- 1. Carregamento e indexação de documentos ---

def carregar_documentos():
    """
    Carrega todas as abas da planilha Excel como documentos para indexação.
    Cada linha do Excel vira um documento de texto com a origem sendo o nome da aba.
    """
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

def _get_index_mtime():
    """Retorna o timestamp da última modificação do índice FAISS."""
    idx_file = Path(INDEX_DIR) / "index.faiss"
    return idx_file.stat().st_mtime if idx_file.exists() else 0

def precisa_reindexar():
    """
    Indica se o índice precisa ser atualizado com base na data de modificação do Excel.
    """
    if not os.path.exists(ARQUIVO_EXCEL):
        logger.warning("Arquivo Excel de canais inexistente.")
        return False
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return True
    last_index = _get_index_mtime()
    last_excel = os.path.getmtime(ARQUIVO_EXCEL)
    return last_excel > last_index

_lock_reindex = threading.Lock()

def indexar_documentos():
    """
    Cria o índice vetorial FAISS com embeddings da OpenAI a partir do Excel.
    Usa chunking por frases para granularidade melhor.
    Indexação automática se o Excel foi atualizado.
    """
    with _lock_reindex:
        try:
            dados = carregar_documentos()
            if not dados:
                raise ValueError("Nenhum conteúdo válido foi encontrado no Excel.")

            docs = [d["content"] for d in dados]
            metadados = ["Origem: " + d["source"] for d in dados]

            from langchain.schema import Document
            documentos = [
                Document(page_content=doc, metadata={"source": meta})
                for doc, meta in zip(docs, metadados)
            ]

            # Chunking por frases (3 sentenças por chunk)
            splitter = NLTKTextSplitter(chunk_size=3, chunk_overlap=1)
            chunks = splitter.split_documents(documentos)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(INDEX_DIR)

            logger.info(f"Indexação concluída com {len(documentos)} documentos e {len(chunks)} chunks.")
            return len(documentos)
        except Exception as e:
            logger.error(f"Erro ao indexar documentos: {e}")
            raise RuntimeError(f"Erro ao indexar documentos: {e}")

def atualizar_se_preciso():
    """
    Atualiza o índice vetorial apenas se o arquivo Excel foi modificado desde a última indexação.
    """
    if precisa_reindexar():
        logger.info("Arquivo Excel alterado. Reindexando conhecimento de canais...")
        return indexar_documentos()
    else:
        logger.info("Índice já está atualizado.")
        return None

def carregar_index():
    """
    Carrega o índice FAISS salvo em disco, garantindo que esteja atualizado.
    """
    atualizar_se_preciso()
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# --- 2. Busca semântica e RAG + IA com cache ---

_resposta_cache = {}

def buscar_conhecimento(pergunta: str, k=5) -> str:
    """
    Realiza uma busca semântica na base indexada e retorna os melhores trechos.
    """
    try:
        db = carregar_index()
        docs = db.similarity_search(pergunta, k=k)
        resposta = "\n\n".join(doc.page_content.strip() for doc in docs)
        return resposta
    except Exception as e:
        logger.error(f"Erro ao buscar conhecimento: {e}")
        return "Não foi possível buscar conhecimento no momento."

from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def buscar_conhecimento_complementado(pergunta: str, k=5) -> str:
    """
    Consulta o RAG e complementa com a OpenAI para respostas mais completas.
    Usa cache para perguntas repetidas.
    """
    cache_key = f"{pergunta}|k={k}"
    if cache_key in _resposta_cache:
        return _resposta_cache[cache_key]

    resposta_rag = buscar_conhecimento(pergunta, k)
    prompt = (
        f"Baseado no conhecimento extraído a seguir, responda à pergunta de maneira clara, prática e adaptada ao mercado brasileiro.\n\n"
        f"Pergunta: {pergunta}\n\n"
        f"Conhecimento extraído:\n{resposta_rag}\n"
    )
    try:
        resposta_ia = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor especialista em canais de vendas B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        resposta_final = (
            f"{resposta_rag}\n\n🔎 Complemento da IA:\n{resposta_ia.choices[0].message.content.strip()}"
        )
        _resposta_cache[cache_key] = resposta_final
        return resposta_final
    except Exception as e:
        logger.warning(f"Erro ao complementar conhecimento com OpenAI: {e}")
        _resposta_cache[cache_key] = resposta_rag
        return resposta_rag

# --- 3. Sugestões e exemplos para integração ---

EXEMPLOS_PERGUNTAS = [
    "Quais são os modelos mais comuns de parceria para empresas SaaS?",
    "Que perfil de canal é ideal para venda de software de gestão?",
    "Qual estrutura de comissionamento é usada em canais de tecnologia?",
    "Quais são os benefícios para parceiros comerciais no segmento industrial?",
    "Como selecionar canais de vendas para soluções B2B?",
    "Diferença entre canal de distribuição e canal de revenda?",
    "O que considerar ao criar um programa de canais?",
]

def exemplos_perguntas():
    """
    Retorna exemplos de perguntas úteis para extração de perfis/modelos de canais.
    """
    return EXEMPLOS_PERGUNTAS

# --- 4. Teste local do módulo ---

if __name__ == "__main__":
    qtd = indexar_documentos()
    print(f"✅ {qtd} registros indexados com sucesso.")
    for pergunta in exemplos_perguntas():
        resultado = buscar_conhecimento_complementado(pergunta)
        print(f"\n📌 PERGUNTA: {pergunta}\nRESPOSTA:\n{resultado}\n")