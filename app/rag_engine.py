from log_config import get_logger
logger = get_logger(__name__)

from openai import OpenAI
import pandas as pd
import numpy as np
import os
import json
import hashlib
from difflib import get_close_matches

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Cache para embeddings (para evitar repetição de chamadas)
embedding_cache_path = "embedding_cache.json"
if os.path.exists(embedding_cache_path):
    with open(embedding_cache_path, "r") as f:
        EMBEDDING_CACHE = json.load(f)
else:
    EMBEDDING_CACHE = {}

def get_embedding(text):
    hash_key = hashlib.sha256(text.encode()).hexdigest()
    if hash_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[hash_key]

    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small")

        embedding = response.data[0].embedding

        EMBEDDING_CACHE[hash_key] = embedding

        # Salvar cache
        with open(embedding_cache_path, "w") as f:
            json.dump(EMBEDDING_CACHE, f)

        return embedding
    except Exception as e:
        logger.error(f"Erro ao gerar embedding: {e}")
        return None

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Carregar base IBGE
df = pd.read_excel("Tabela 14.xlsx")
df.columns = [
    "Municipio", "Codigo_CNAE", "Descricao_CNAE", "Unidades_Locais",
    "Pessoal_Total", "Pessoal_Assalariado", "Assalariado_Medio",
    "Remuneracao_Mil_R$", "Salario_Medio_SM", "Salario_Medio_R$"
]

df = df[df["Municipio"].notna()]
df = df[~df["Municipio"].astype(str).str.contains("Municípios com|Tabela|Total", na=False)]
df = df[~df["Unidades_Locais"].astype(str).isin(["-", "nan"])]
df = df[df["Salario_Medio_R$"].astype(str).str.replace(",", "").str.replace(".", "").str.isnumeric()]
df["Unidades_Locais"] = pd.to_numeric(df["Unidades_Locais"], errors="coerce")
df["Salario_Medio_R$"] = pd.to_numeric(df["Salario_Medio_R$"], errors="coerce")

def simular_populacao_pib(df_segmento):
    municipios = df_segmento["Municipio"].unique()
    empresas = df_segmento.groupby("Municipio")["Unidades_Locais"].sum().values
    populacoes = np.clip(np.round(empresas * np.random.uniform(15, 60)).astype(int), 1000, 20_000_000)
    pibs = np.clip(np.round(empresas * np.random.uniform(0.02, 0.08), 2), 0.3, 200.0)
    perfil_canal = np.round(empresas * np.random.uniform(0.1, 0.5)).astype(int)
    media_salarial_base = df_segmento.groupby("Municipio")["Salario_Medio_R$"].mean().values
    salarios = np.clip(np.round(media_salarial_base * np.random.uniform(0.85, 1.25), 2), 1500.0, 15000.0)

    return pd.DataFrame({
        "Municipio": municipios,
        "Populacao": populacoes,
        "PIB": pibs,
        "Empresas_Segmento": empresas,
        "Empresas_Perfil_Canal": perfil_canal,
        "Salario_Medio_R$": salarios
    })

def normalizar_segmentos(segmentos: str):
    if isinstance(segmentos, list):
        segmentos = " ".join(segmentos)
    return [s.strip() for s in segmentos.replace(",", " ").split() if len(s.strip()) > 2]

def buscar_similares_embedding(termo, descricoes, threshold=0.85):
    try:
        termo_emb = get_embedding(termo)
        if termo_emb is None:
            return termo
        scores = [
            (descricao, cosine_similarity(termo_emb, get_embedding(descricao)))
            for descricao in descricoes
        ]
        melhor_match = sorted(scores, key=lambda x: x[1], reverse=True)[0]
        return melhor_match[0] if melhor_match[1] >= threshold else termo
    except Exception as e:
        logger.error(f"Erro em similaridade por embedding: {e}")
        return termo

def buscar_cidades_na_openai(segmentos: list[str], cidades_existentes: list[str], faltantes: int):
    prompt = f"""
Considere segmentos de atuação: {", ".join(segmentos)}.
Com base nisso, sugira {faltantes} cidades brasileiras com grande potencial de mercado para empresas desses segmentos.
Evite repetir as cidades já listadas: {", ".join(cidades_existentes)}.
Liste apenas os nomes das cidades, em uma única linha, separados por vírgula.
"""
    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um especialista em inteligência de mercado regional brasileiro."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        cidades_sugeridas = resposta['choices'][0]['message']['content']
        return [c.strip() for c in cidades_sugeridas.split(",") if c.strip()]
    except Exception as e:
        logger.error(f"Erro ao buscar cidades com OpenAI: {e}")
        return []

def filtrar_municipios_por_segmentos_multiplos(segmentos: str, top_n: int = 30, ordenar_por=None):
    if ordenar_por is None:
        ordenar_por = ["Empresas_Segmento", "Salario_Medio_R$"]

    segmentos_lista = normalizar_segmentos(segmentos)
    logger.info(f"Segmentos identificados para busca: {segmentos_lista}")

    try:
        filtrados = pd.DataFrame()
        descricoes_cnae = df["Descricao_CNAE"].dropna().unique().tolist()

        for termo in segmentos_lista:
            termo_similar = buscar_similares_embedding(termo, descricoes_cnae)
            encontrados = df[df["Descricao_CNAE"].astype(str).str.contains(termo_similar, case=False, na=False)]
            if not encontrados.empty:
                filtrados = pd.concat([filtrados, encontrados])
            else:
                logger.warning(f"Segmento '{termo}' nao encontrado no RAG.")

        if filtrados.empty:
            return pd.DataFrame(columns=[
                "Municipio", "Populacao", "PIB",
                "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"
            ])

        for col in ["Municipio", "Populacao", "PIB", "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"]:
            if col not in final_df.columns:
                if col == "Municipio":
                    final_df[col] = "CidadeDesconhecida"
                else:
                    final_df[col] = 0
        
        
        dados_complementares = simular_populacao_pib(filtrados)
        final_df = dados_complementares.groupby("Municipio").sum(numeric_only=True).reset_index()

        if "Salario_Medio_R$" not in final_df:
            final_df["Salario_Medio_R$"] = np.random.uniform(2000, 8000, size=len(final_df))

        final_df = final_df.sort_values(by=ordenar_por, ascending=False).reset_index(drop=True)

        if len(final_df) < top_n:
            faltam = top_n - len(final_df)
            cidades_existentes = final_df["Municipio"].tolist()
            sugestoes_openai = buscar_cidades_na_openai(segmentos_lista, cidades_existentes, faltam)

            if sugestoes_openai:
                extras = pd.DataFrame({
                    "Municipio": sugestoes_openai,
                    "Populacao": np.random.randint(1000, 20000000, size=len(sugestoes_openai)),
                    "PIB": np.round(np.random.uniform(0.3, 10.0, size=len(sugestoes_openai)), 2),
                    "Empresas_Segmento": np.random.randint(20, 100, size=len(sugestoes_openai)),
                    "Empresas_Perfil_Canal": np.random.randint(5, 50, size=len(sugestoes_openai)),
                    "Salario_Medio_R$": np.round(np.random.uniform(1800, 3500, size=len(sugestoes_openai)), 2)
                })
                final_df = pd.concat([final_df, extras], ignore_index=True)

        return final_df.head(top_n)

    except Exception as e:
        logger.error(f"Erro ao processar segmentos '{segmentos}': {e}")
        return pd.DataFrame(columns=[
            "Municipio", "Populacao", "PIB",
            "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"
        ])

def gerar_tabela_html(dataframe):
    if dataframe.empty:
        return """
        <div class='paragrafo'>
            <h3 style='color:#5e17eb;'>📍 Nenhum município encontrado para este segmento.</h3>
            <p>Revise o segmento informado ou tente outro termo mais genérico.</p>
        </div>
        """
    linhas = ""
    for _, row in dataframe.iterrows():
        linhas += (
            f"<tr>"
            f"<td>{row['Municipio']}</td>"
            f"<td>{row['Populacao']:,}</td>"
            f"<td>R${row['PIB']:.2f} bi</td>"
            f"<td>{int(row['Empresas_Segmento'])}</td>"
            f"<td>{int(row['Empresas_Perfil_Canal'])}</td>"
            f"<td>R${row['Salario_Medio_R$']:.2f}</td>"
            f"</tr>"
        )

    return f"""
    <div class='paragrafo'>
        <h3 style='color:#5e17eb;'>📍 Top 30 Municípios com Maior Potencial para Canais</h3>
        <table border='0' width='100%' style='font-size:15px; line-height:1.5; border-collapse:collapse; margin-top:15px;'>
            <thead style='background:#f0f0f0;'>
                <tr>
                    <th align='left'>Município</th>
                    <th align='left'>População</th>
                    <th align='left'>PIB</th>
                    <th align='left'>Empresas no Segmento</th>
                    <th align='left'>Empresas com Perfil de Canal</th>
                    <th align='left'>Salário Médio (R$)</th>
                </tr>
            </thead>
            <tbody>
                {linhas}
            </tbody>
        </table>
    </div>
    """
