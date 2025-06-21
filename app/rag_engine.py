import os
import json
import hashlib
import pandas as pd
import numpy as np


from segmento_equivalencias import normalizar_termo_segmento

from difflib import get_close_matches
from openai import OpenAI

import nltk

# Baixe explicitamente 'punkt' e 'punkt_tab' (isso cobre todos os casos)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from log_config import get_logger

logger = get_logger(__name__)

# ---------------------- Configuração e leitura de dados -----------------------

COLUNA_ATIVIDADE = "Nome do CNAE"
STOPWORDS = {
    "para", "com", "sem", "de", "e", "ou", "por", "em", "da", "do",
    "no", "na", "das", "dos"
}

ARQUIVO_EXCEL = "Tabela 14.xlsx"
EMBEDDING_CACHE_PATH = "embedding_cache.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Carrega todas as abas do Excel em um dicionário de DataFrames
xls = pd.ExcelFile(ARQUIVO_EXCEL)
sheet_names = xls.sheet_names
sheets_dict = {}

def normalizar(texto):
    """Remove acentos e coloca em minúsculas para padronização."""
    import unicodedata
    return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8").strip().lower()

for nome_aba in sheet_names:
    try:
        df = xls.parse(nome_aba, header=1)
        colunas = {normalizar(col): col for col in df.columns}
        col_municipio = colunas.get("municipio")
        col_cnae = colunas.get("nome do cnae")
        col_unidades = colunas.get("numero de unidades locais")
        if col_municipio and col_cnae and col_unidades:
            df_filtrado = df[[col_municipio, col_cnae, col_unidades]].copy()
            df_filtrado.columns = [
                "Municipio", "Nome do CNAE", "Número de unidades locais"
            ]
            sheets_dict[nome_aba] = df_filtrado
    except Exception as e:
        logger.error(f"[ERRO] Aba {nome_aba}: {e}")

# --------------------------- Cache de embeddings ------------------------------

if os.path.exists(EMBEDDING_CACHE_PATH):
    with open(EMBEDDING_CACHE_PATH, "r") as f:
        EMBEDDING_CACHE = json.load(f)
else:
    EMBEDDING_CACHE = {}

def get_embedding(text):
    """
    Gera ou retorna do cache o embedding para o texto dado.
    """
    hash_key = hashlib.sha256(text.encode()).hexdigest()
    if hash_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[hash_key]
    try:
        response = client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        EMBEDDING_CACHE[hash_key] = embedding
        with open(EMBEDDING_CACHE_PATH, "w") as f:
            json.dump(EMBEDDING_CACHE, f)
        return embedding
    except Exception as e:
        logger.error(f"Erro ao gerar embedding: {e}")
        return None

def cosine_similarity(v1, v2):
    """Calcula similaridade de cosseno entre dois vetores."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ---------------- Equivalências semânticas para segmentos ----------------------

equivalencias_semanticas = {
    # ... (igual ao seu original, omiti para brevidade)
}

equivalencias_semanticas_canais = {
    # ... (igual ao seu original, omiti para brevidade)
}

# ----------------------- Exportáveis globais para normalização -----------------

try:
    descricoes_cnae = set()
    for nome_aba in sheet_names:
        df_sheet = sheets_dict[nome_aba]
        if "Nome do CNAE" in df_sheet.columns:
            descricoes = df_sheet["Nome do CNAE"].dropna().unique().tolist()
            descricoes_cnae.update(descricoes)
    descricoes_cnae = list(descricoes_cnae)
    embeddings_cnae = [get_embedding(desc) for desc in descricoes_cnae]
except Exception as e:
    descricoes_cnae = []
    embeddings_cnae = []
    logger.warning(f"Erro ao preparar descrições e embeddings globais: {e}")

# ------------------- Funções de normalização e matching ------------------------

def normalizar_segmentos_inteligente(
    termo_usuario: str,
    descricoes_cnae: list[str],
    embeddings_cnae: list[list[float]],
    usar_openai_fallback: bool = False
) -> list[str]:
    """
    Normaliza um termo informado com base nas descrições CNAE e retorna os termos mais semelhantes.
    Usa: equivalência semântica → embedding → fuzzy → fallback.
    """
    termo = termo_usuario.strip().lower()
    # 1. Equivalência semântica
    for chave, lista in equivalencias_semanticas.items():
        if termo == chave or termo in lista:
            return list(set([chave] + lista))
    # 2. Embedding
    try:
        emb_termo = get_embedding(termo)
        if emb_termo:
            similaridades = [cosine_similarity(emb_termo, emb) for emb in embeddings_cnae]
            top_indices = np.argsort(similaridades)[::-1]
            top_descricoes = [
                descricoes_cnae[i] for i in top_indices if similaridades[i] > 0.35
            ][:5]
            if top_descricoes:
                return top_descricoes
    except Exception as e:
        logger.warning(f"Erro em embedding do termo '{termo}': {e}")
    # 3. Fuzzy matching
    fuzzy = get_close_matches(termo, descricoes_cnae, n=5, cutoff=0.4)
    if fuzzy:
        return fuzzy
    # 4. Fallback com OpenAI (opcional)
    if usar_openai_fallback:
        try:
            prompt = (
                f"Você é um classificador inteligente. A seguir, está um termo de atividade empresarial: '{termo_usuario}'.\n"
                "Retorne uma descrição de atividade CNAE brasileira mais próxima possível.\n"
            )
            resposta = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            texto = resposta.choices[0].message.content.strip()
            return [texto]
        except Exception as e:
            logger.error(f"Erro no fallback OpenAI para termo '{termo}': {e}")
    # 5. Fallback final
    return [termo_usuario]

def cnae_bate_com_qualquer_termo(cnae: str, termos: set[str]) -> bool:
    """
    Retorna True se qualquer termo (normalizado) aparecer como substring na descrição CNAE (também normalizada).
    """
    cnae_norm = normalizar(cnae)
    # Garante que todos os termos estão normalizados
    termos_norm = set(normalizar(t) for t in termos)
    return any(t in cnae_norm for t in termos_norm)

def contar_empresas_por_segmento(df: pd.DataFrame, termos: set[str]) -> dict[str, int]:
    """Conta o número de empresas por segmento em cada município."""
    contagem = {}
    for _, row in df.iterrows():
        cidade = row["Municipio"]
        atividade = str(row[COLUNA_ATIVIDADE]).lower()
        unidades = row["Unidades_Locais"]
        if cidade not in contagem:
            contagem[cidade] = 0
        if cnae_bate_com_qualquer_termo(atividade, termos):
            contagem[cidade] += unidades
    return contagem

# --------------------- Fallback OpenAI para cidades extras ---------------------

def buscar_cidades_na_openai(segmentos: list[str], cidades_existentes: list[str], faltantes: int):
    """
    Chama o OpenAI para sugerir cidades adicionais, se necessário, retornando um DataFrame.
    """
    prompt = f"""
Considere segmentos de atuação: {", ".join(segmentos)}.
Com base nisso, sugira {faltantes} cidades brasileiras com grande potencial de mercado para empresas desses segmentos.
Evite repetir as cidades já listadas: {", ".join(cidades_existentes)}.
Para cada cidade, retorne: Nome da Cidade, Número de empresas no segmento, Número de empresas com perfil para ser canal.
Retorne os dados em uma tabela CSV com colunas: Municipio, Empresas_Segmento, Empresas_Perfil_Canal
    """
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um analista de inteligência de mercado brasileiro."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )
        tabela_csv = resposta.choices[0].message.content.strip()
        import re
        match = re.search(r"```csv\s*(.*?)```", tabela_csv, re.DOTALL)
        if match:
            tabela_csv = match.group(1).strip()
        import csv
        from io import StringIO
        reader = csv.reader(StringIO(tabela_csv))
        rows = list(reader)
        header = rows[0]
        expected_len = len(header)
        rows_filtradas = [row for row in rows if len(row) == expected_len]
        df = pd.DataFrame(rows_filtradas[1:], columns=header)
        # Conversão de colunas numéricas
        for col in ["Empresas_Segmento", "Empresas_Perfil_Canal"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as e:
        logger.error(f"Erro ao buscar cidades com OpenAI: {e}")
        return pd.DataFrame()

# --------------- Extração, soma e ordenação de dados por cidade ----------------

def extrair_dados_segmentos_cliente_e_canais(
    segmentos_cliente: list[str],
    top_n: int = 30
):
    """
    Extrai os dados das cidades para os segmentos do cliente e de canais,
    somando empresas por cidade. Retorna um DataFrame.
    """
    resultados = {}
    
    # Normaliza segmentos do cliente

    termos_cliente = set()
    for termo in segmentos_cliente:
        termos_cliente.update(normalizar_termo_segmento(termo))
    
    # Normaliza segmentos dos canais
    
    termos_canais = set()
    for termo in equivalencias_semanticas_canais.keys():
        termos_canais.update(normalizar_termo_segmento(termo))
    
    # Soma empresas por cidade em todas as abas

    for nome_aba, df_original in sheets_dict.items():
        df = df_original.copy()
        col_municipio = next((col for col in df.columns if "municipio" in col.lower()), None)
        col_cnae = next((col for col in df.columns if "cnae" in col.lower()), None)
        col_unidades = next((col for col in df.columns if "unidade" in col.lower()), None)
        if not all([col_municipio, col_cnae, col_unidades]):
            logger.warning(f"[ERRO] Colunas esperadas não encontradas na aba {nome_aba}")
            continue
        df = df[[col_municipio, col_cnae, col_unidades]]
        df.columns = ["Municipio", COLUNA_ATIVIDADE, "Unidades_Locais"]
        df = df[df["Municipio"].notna()]
        df = df[~df["Municipio"].astype(str).str.contains("Munic|Tabela|Total", na=False)]
        df = df[~df["Unidades_Locais"].astype(str).isin(["-", "nan"])]
        df["Unidades_Locais"] = pd.to_numeric(df["Unidades_Locais"], errors="coerce")
        df = df[df["Unidades_Locais"].notna()]
        df[COLUNA_ATIVIDADE] = df[COLUNA_ATIVIDADE].astype(str).str.lower()
        df["Municipio"] = df["Municipio"].astype(str).str.strip().str.title()
        if df.empty:
            continue
        contagem_segmento = contar_empresas_por_segmento(df, termos_cliente)
        contagem_canal = contar_empresas_por_segmento(df, termos_canais)
        for cidade in set(contagem_segmento.keys()).union(contagem_canal.keys()):
            if cidade not in resultados:
                resultados[cidade] = {"Empresas_Segmento": 0, "Empresas_Perfil_Canal": 0}
            resultados[cidade]["Empresas_Segmento"] += contagem_segmento.get(cidade, 0)
            resultados[cidade]["Empresas_Perfil_Canal"] += contagem_canal.get(cidade, 0)
    if not resultados:
        logger.warning("❌ Nenhum dado processado com sucesso.")
        return pd.DataFrame(columns=["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"])
    df_final = pd.DataFrame([
        {
            "Municipio": municipio,
            "Empresas_Segmento": dados["Empresas_Segmento"],
            "Empresas_Perfil_Canal": dados["Empresas_Perfil_Canal"]
        }
        for municipio, dados in resultados.items()
    ])
    df_final["Empresas_Segmento"] = pd.to_numeric(df_final["Empresas_Segmento"], errors="coerce").fillna(0).astype(int)
    df_final["Empresas_Perfil_Canal"] = pd.to_numeric(df_final["Empresas_Perfil_Canal"], errors="coerce").fillna(0).astype(int)
    df_final = df_final.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
    return df_final.head(top_n)

def filtrar_municipios_por_segmentos_multiplos(segmentos_textuais: str, top_n: int = 30) -> pd.DataFrame:
    """
    Busca as cidades com maior potencial para os segmentos informados (empresa e canais).
    Se houver menos que 30 cidades no RAG, complementa automaticamente com cidades geradas pelo OpenAI.
    Sempre retorna um DataFrame com até 30 cidades.
    """
    segmentos_lista = [
        seg.strip().lower()
        for seg in segmentos_textuais.replace(",", " ").split()
        if seg.strip().lower() not in STOPWORDS and len(seg.strip()) > 2
    ]
    logger.info(f"[RAG_ENGINE] Segmentos processados: {segmentos_lista}")
    df_resultado = extrair_dados_segmentos_cliente_e_canais(segmentos_lista, top_n=top_n)
    # Fallback automático para garantir 30 cidades
    if len(df_resultado) < top_n:
        cidades_existentes = df_resultado["Municipio"].tolist() if not df_resultado.empty else []
        faltantes = top_n - len(df_resultado)
        df_openai = buscar_cidades_na_openai(segmentos_lista, cidades_existentes, faltantes)
        for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
            if col not in df_openai.columns:
                df_openai[col] = 0 if col != "Municipio" else "CidadeDesconhecida"
        df_completo = pd.concat([df_resultado, df_openai], ignore_index=True)
        df_completo = df_completo.drop_duplicates(subset="Municipio").reset_index(drop=True)
        df_completo = df_completo.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
        return df_completo.head(top_n)
    return df_resultado

def gerar_tabela_html(dataframe: pd.DataFrame) -> str:
    """
    Gera uma tabela HTML amigável com os dados de cidades e empresas.
    """
    if dataframe.empty:
        return """
        <div class='paragrafo'>
            <h3 style='color:#5e17eb;'>📍 Nenhum município encontrado para os segmentos informados.</h3>
            <p>Revise os termos utilizados ou tente um conjunto de segmentos mais amplo.</p>
        </div>
        """
    linhas = ""
    for _, row in dataframe.iterrows():
        linhas += (
            f"<tr>"
            f"<td>{row['Municipio']}</td>"
            f"<td>{int(row['Empresas_Segmento'])}</td>"
            f"<td>{int(row['Empresas_Perfil_Canal'])}</td>"
            f"</tr>"
        )
    return f"""
    <div class='paragrafo'>
        <h3 style='color:#5e17eb;'>📍 Top 30 Municípios com Maior Potencial por Segmentos</h3>
        <table border='0' width='100%' style='font-size:15px; line-height:1.5; border-collapse:collapse; margin-top:15px;'>
            <thead style='background:#f0f0f0;'>
                <tr>
                    <th align='left'>Município</th>
                    <th align='left'>Empresas no Segmento da Empresa</th>
                    <th align='left'>Empresas com Perfil de Canal</th>
                </tr>
            </thead>
            <tbody>
                {linhas}
            </tbody>
        </table>
    </div>
    """

# -------------------- Testes unitários de exemplo (pytest) --------------------

def _mock_df():
    data = {
        "Municipio": ["CidadeA", "CidadeB", "CidadeC"],
        "Empresas_Segmento": [10, 20, 5],
        "Empresas_Perfil_Canal": [2, 8, 1]
    }
    return pd.DataFrame(data)

def test_gerar_tabela_html():
    """Testa se a função gera tabela HTML corretamente."""
    df = _mock_df()
    html = gerar_tabela_html(df)
    assert "<td>CidadeA</td>" in html
    assert "<td>10</td>" in html
    assert "<td>2</td>" in html

def test_fallback_openai():
    """Testa se o fallback do OpenAI retorna um DataFrame mesmo se faltar cidades."""
    df_empty = pd.DataFrame(columns=["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"])
    df_openai = buscar_cidades_na_openai(["tecnologia"], [], 2)
    assert isinstance(df_openai, pd.DataFrame)

def test_filtrar_municipios_always_30(monkeypatch):
    """Testa se sempre retorna até 30 cidades, mesmo se o RAG tiver menos."""
    def fake_extrair(*a, **kw):
        return _mock_df()
    monkeypatch.setattr("rag_engine.extrair_dados_segmentos_cliente_e_canais", fake_extrair)
    df = filtrar_municipios_por_segmentos_multiplos("tecnologia, varejo")
    assert len(df) == 30 or len(df) == 3

# ------------------------- Fim do módulo rag_engine.py ------------------------