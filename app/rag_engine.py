from log_config import get_logger
logger = get_logger(__name__)

import openai
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import json
import hashlib
from difflib import get_close_matches

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COLUNA_ATIVIDADE = "Seções e divisões da classificação de atividades"

# Cache para embeddings
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

# Lê todas as abas do arquivo Excel

sheets_dict = pd.read_excel("Tabela 14.xlsx", sheet_name=None)
sheet_names = list(sheets_dict.keys())

raw_df.rename(columns={
    "Seções e divisões da classificação de atividades": "Descricao_CNAE"
}, inplace=True)

raw_df.columns = [
    "Municipio", "Codigo_CNAE", "Seções e divisões da classificação de atividades",
    "Unidades_Locais", "Pessoal_Total", "Pessoal_Assalariado", "Assalariado_Medio",
    "Remuneracao_Mil_R$", "Salario_Medio_SM", "Salario_Medio_R$"
]


raw_df = raw_df[raw_df["Municipio"].notna()]
raw_df = raw_df[~raw_df["Municipio"].astype(str).str.contains("Munic|Tabela|Total", na=False)]
raw_df = raw_df[~raw_df["Unidades_Locais"].astype(str).isin(["-", "nan"])]
raw_df = raw_df[raw_df["Salario_Medio_R$"].astype(str).str.replace(",", "").str.replace(".", "").str.isnumeric()]
raw_df["Unidades_Locais"] = pd.to_numeric(raw_df["Unidades_Locais"], errors="coerce")
raw_df["Salario_Medio_R$"] = pd.to_numeric(raw_df["Salario_Medio_R$"], errors="coerce")

STOPWORDS = {"para", "com", "sem", "de", "e", "ou", "por", "em", "da", "do", "no", "na", "das", "dos"}

equivalencias_semanticas = {
    "agronegócio": ["agropecuária", "agricultura", "pecuária", "produção rural", "cultivo", "plantação", "fazenda"],
    "alimentício": ["alimentos", "indústria de alimentos", "bebidas", "fabricação de alimentos", "comida", "processamento alimentar"],
    "logística": ["transporte", "armazenagem", "distribuição", "frete", "entrega", "supply chain", "cadeia logística"],
    "varejo": ["comércio varejista", "lojas", "shopping", "ponto de venda", "pdv", "retail", "comércio"],
    "atacado": ["comércio atacadista", "distribuidor", "distribuição em massa", "atacadista"],
    "farmacêutico": ["remédios", "medicamentos", "farmácias", "indústria farmacêutica", "saúde", "laboratório"],
    "hospitalar": ["hospitais", "clínicas", "saúde", "unidade de saúde", "UPA", "postos de saúde", "pronto socorro"],
    "construção civil": ["obras", "empreiteira", "construtora", "infraestrutura", "engenharia civil", "imóveis em construção"],
    "imobiliário": ["imóveis", "incorporadora", "construtora", "venda de imóveis", "locação de imóveis"],
    "financeiro": ["finanças", "bancos", "meios de pagamento", "instituições financeiras", "cooperativa de crédito"],
    "seguros": ["corretora", "plano de saúde", "seguradora", "seguro de vida", "auto", "patrimônio"],
    "automotivo": ["carros", "veículos", "oficinas", "autopeças", "montadoras", "revendedoras"],
    "educação": ["escolas", "ensino", "universidade", "colégio", "faculdade", "instituições de ensino"],
    "tecnologia": ["TI", "software", "hardware", "sistemas", "desenvolvimento de software", "startups"],
    "indústria têxtil": ["tecidos", "malharia", "confecção", "roupas", "vestuário", "moda"],
    "calçadista": ["sapatos", "calçados", "fabricação de calçados"],
    "cosméticos": ["beleza", "perfumes", "estética", "cuidados pessoais", "produtos de beleza"],
    "mineração": ["mineradora", "extração mineral", "bauxita", "ferro", "minério", "carvão"],
    "siderurgia": ["aço", "metalurgia", "fundições", "laminação de aço", "indústria do aço"],
    "químico": ["produtos químicos", "solventes", "resinas", "indústria química"],
    "plástico": ["indústria plástica", "embalagens plásticas", "injetoras", "extrusoras"],
    "embalagens": ["packaging", "caixas", "rótulos", "frascos", "embalagens em geral"],
    "papel e celulose": ["papel", "indústria de papel", "fábricas de papel", "papelão", "celulose"],
    "editorial": ["gráfica", "editoras", "livros", "publicações", "revistas", "jornais"],
    "energia": ["usinas", "distribuidoras de energia", "solar", "eólica", "hidrelétrica", "geração de energia"],
    "telecomunicações": ["telefonia", "internet", "provedores", "infraestrutura de redes", "operadoras"],
    "limpeza": ["produtos de limpeza", "higiene", "desinfetantes", "sanitização"],
    "condomínios": ["síndico", "gestão condominial", "residenciais", "condomínios empresariais"],
    "hotelaria": ["hotéis", "resorts", "pousadas", "turismo", "hospitalidade"],
    "turismo": ["agências", "viagens", "pacotes turísticos", "guias turísticos"],
    "transportes": ["fretamento", "rodoviário", "ferroviário", "marítimo", "logística", "entregas"],
    "aeronáutico": ["aviões", "manutenção de aeronaves", "aeroportos", "aeronaves", "fabricantes de aeronaves"],
    "naval": ["indústria naval", "embarcações", "estaleiros", "transporte marítimo"],
    "mecânico": ["usinagem", "autopeças", "componentes mecânicos", "mecânica industrial"],
    "metalúrgico": ["fundições", "soldagem", "indústria de metais", "usinagem"],
    "moveleiro": ["móveis", "marcenaria", "indústria de móveis", "design de interiores"],
    "frigorífico": ["carnes", "processamento de alimentos", "abatedouros", "resfriados"],
    "bebidas": ["cervejarias", "refrigerantes", "água", "vinhos", "indústria de bebidas"],
    "meio ambiente": ["resíduos", "coleta seletiva", "tratamento de água", "energia renovável", "reciclagem"],
    "segurança": ["monitoramento", "portaria", "segurança patrimonial", "vigilância", "alarmistas"],
    "RH": ["recrutamento", "recursos humanos", "terceirização de mão de obra", "gestão de talentos"],
    "jurídico": ["advocacia", "escritório de advocacia", "consultoria jurídica"],
    "contábil": ["contabilidade", "escritórios contábeis", "consultoria tributária"],
    "esportes": ["academias", "fitness", "esporte coletivo", "esporte individual", "clubes"],
    "entretenimento": ["eventos", "shows", "cinema", "música", "streaming"],
    "e-commerce": ["lojas online", "marketplaces", "plataformas de venda", "comércio eletrônico"],
    "pet": ["produtos para animais", "veterinários", "clínicas pet", "alimentos para pets"],
    "eventos": ["cerimonial", "buffets", "organização de eventos", "festas", "congressos"],
    "limpeza urbana": ["coleta de lixo", "varrição", "gestão de resíduos urbanos", "serviços públicos"],
    "serviços gerais": ["terceirização", "multisserviços", "facilities", "mão de obra auxiliar"]
}

def normalizar_segmentos(segmentos: str):
    if isinstance(segmentos, list):
        segmentos = " ".join(segmentos)
    return [s.strip() for s in segmentos.replace(",", " ").split() if len(s.strip()) > 2 and s.strip().lower() not in STOPWORDS]

def simular_populacao_pib(df_segmento):
    municipios = df_segmento["Municipio"].unique()
    empresas = df_segmento.groupby("Municipio")["Unidades_Locais"].sum().values
    populacoes = np.clip(np.round(empresas * np.random.uniform(15, 60)).astype(int), 1000, 20_000_000)
    pibs = np.clip(np.round(empresas * np.random.uniform(0.02, 0.08), 2), 0.3, 200.0)
    perfil_canal = np.round(empresas * np.random.uniform(0.1, 0.5)).astype(int)
    media_salarial_base = df_segmento.groupby("Municipio")["Salario_Medio_R$"].mean().values
    salarios = np.clip(np.round(media_salarial_base * np.random.uniform(0.85, 1.25), 2), 1500.0, 15000.0)

def processar_aba_por_segmento(df_sheet, segmentos_lista):
    resultados = pd.DataFrame()
    for termo in segmentos_lista:
        encontrados = df_sheet[df_sheet[COLUNA_ATIVIDADE].astype(str).str.lower().str.contains(termo.lower(), na=False)]
        if not encontrados.empty:
            resultados = pd.concat([resultados, encontrados])
        else:
            logger.warning(f"Segmento '{termo}' não encontrado na aba.")
    return resultados


    return pd.DataFrame({
        "Municipio": municipios,
        "Populacao": populacoes,
        "PIB": pibs,
        "Empresas_Segmento": empresas,
        "Empresas_Perfil_Canal": perfil_canal,
        "Salario_Medio_R$": salarios
    })

def buscar_similares_embedding(termo, descricoes, threshold=0.35):
    try:
        termo_emb = get_embedding(termo)
        if termo_emb is None:
            return termo

        scores = [
            (descricao, cosine_similarity(termo_emb, get_embedding(descricao)))
            for descricao in descricoes
        ]

        melhor_match = sorted(scores, key=lambda x: x[1], reverse=True)[0]
        if melhor_match[1] >= threshold:
            return melhor_match[0]

        logger.warning(f"Baixa similaridade para termo '{termo}': similaridade {melhor_match[1]:.4f}")
        alternativas = get_close_matches(termo, descricoes, n=1, cutoff=0.6)
        if alternativas:
            return alternativas[0]

        return termo
    except Exception as e:
        logger.error(f"Erro em similaridade por embedding: {e}")
        return termo

def normalizar_segmentos_inteligente(termo_usuario, descricoes_cnae, embeddings_cnae):
    termo = termo_usuario.strip().lower()

    # 1. Substituição via dicionário
    for chave, lista in equivalencias_semanticas.items():
        if termo in [chave] + lista:
            return lista

    # 2. Embedding semântico
    try:
        emb_termo = get_embedding(termo)
        similaridades = [np.dot(emb_termo, e) for e in embeddings_cnae]
        top_indices = np.argsort(similaridades)[::-1][:5]
        top_descricoes = [descricoes_cnae[i] for i in top_indices if similaridades[i] > 0.30]
        if not top_descricoes:
            top_descricoes = [descricoes_cnae[top_indices[0]]]

        if top_descricoes:
            return top_descricoes
    except:
        pass

    # 3. Fuzzy Matching
    fuzzy = get_close_matches(termo, descricoes_cnae, n=3, cutoff=0.30)
    if fuzzy:
        return fuzzy

    # 4. Fallback OpenAI
    try:
        resposta = perguntar_para_openai(termo)
        return [resposta]
    except:
        return [termo]


def buscar_cidades_na_openai(segmentos: list[str], cidades_existentes: list[str], faltantes: int):
    prompt = f"""
Considere segmentos de atuação: {", ".join(segmentos)}.
Com base nisso, sugira {faltantes} cidades brasileiras com grande potencial de mercado para empresas desses segmentos.
Evite repetir as cidades já listadas: {", ".join(cidades_existentes)}.
Para cada cidade, retorne: Nome da Cidade, Estado, População, PIB estimado, Número de empresas no segmento, Número de empresas com perfil para ser canal.
Retorne os dados em uma tabela CSV com colunas: Municipio, Estado, Populacao, PIB, Empresas_Segmento, Empresas_Perfil_Canal
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
        import io
        tabela_csv = resposta.choices[0].message.content.strip()
        df = pd.read_csv(io.StringIO(tabela_csv))
        return df
    except Exception as e:
        logger.error(f"Erro ao buscar cidades com OpenAI: {e}")
        return pd.DataFrame()


def filtrar_municipios_por_segmentos_multiplos(segmentos: str, top_n: int = 30):

    descricoes_cnae = raw_df[COLUNA_ATIVIDADE].dropna().unique().tolist()
    embeddings_cnae = [get_embedding(desc) for desc in descricoes_cnae]
    segmentos_lista = normalizar_segmentos_inteligente(segmentos, descricoes_cnae, embeddings_cnae)

    try:
        filtrados = pd.DataFrame()
        descricoes_cnae = raw_df[COLUNA_ATIVIDADE].dropna().unique().tolist()
        embeddings_cnae = [get_embedding(desc) for desc in descricoes_cnae]
        segmentos_lista = normalizar_segmentos_inteligente(segmentos, descricoes_cnae, embeddings_cnae)

        logger.info(f"Segmentos identificados para busca: {segmentos_lista}")


        filtrados = pd.DataFrame()

        for nome_aba in sheet_names:
            df_sheet = sheets_dict[nome_aba]

            df_sheet.rename(columns={
            "Seções e divisões da classificação de atividades": COLUNA_ATIVIDADE}, inplace=True)

            df_sheet.columns = [
                "Municipio", "Codigo_CNAE", COLUNA_ATIVIDADE,
                "Unidades_Locais", "Pessoal_Total", "Pessoal_Assalariado", "Assalariado_Medio",
                "Remuneracao_Mil_R$", "Salario_Medio_SM", "Salario_Medio_R$"
            ]

            df_sheet = df_sheet[df_sheet["Municipio"].notna()]
            df_sheet = df_sheet[~df_sheet["Municipio"].astype(str).str.contains("Munic|Tabela|Total", na=False)]
            df_sheet = df_sheet[~df_sheet["Unidades_Locais"].astype(str).isin(["-", "nan"])]
            df_sheet = df_sheet[df_sheet["Salario_Medio_R$"].astype(str).str.replace(",", "").str.replace(".", "").str.isnumeric()]
            df_sheet["Unidades_Locais"] = pd.to_numeric(df_sheet["Unidades_Locais"], errors="coerce")
            df_sheet["Salario_Medio_R$"] = pd.to_numeric(df_sheet["Salario_Medio_R$"], errors="coerce")

            resultado_aba = processar_aba_por_segmento(df_sheet, segmentos_lista)

        if not resultado_aba.empty:
            filtrados = pd.concat([filtrados, resultado_aba])

        if filtrados.empty:
            return pd.DataFrame(columns=["Municipio", "Populacao", "PIB", "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"])

        dados_complementares = simular_populacao_pib(filtrados)
        final_df = dados_complementares.groupby("Municipio").sum(numeric_only=True).reset_index()
        final_df = final_df.sort_values(by="Empresas_Segmento", ascending=False).head(top_n)

        if len(final_df) < top_n:
            cidades_existentes = final_df["Municipio"].tolist()
            faltam = top_n - len(final_df)
            logger.warning(f"Apenas {len(final_df)} cidades encontradas no RAG. Buscando {faltam} na OpenAI.")
            sugestoes_openai = buscar_cidades_na_openai(segmentos_lista, cidades_existentes, faltam)

            if not sugestoes_openai.empty:
                colunas_necessarias = ["Municipio", "Populacao", "PIB", "Empresas_Segmento", "Empresas_Perfil_Canal"]
                for coluna in colunas_necessarias:
                    if coluna not in sugestoes_openai.columns:
                        sugestoes_openai[coluna] = 0

                if "Salario_Medio_R$" not in sugestoes_openai.columns:
                    sugestoes_openai["Salario_Medio_R$"] = 5500.0  # valor médio estimado

                extras = sugestoes_openai[colunas_necessarias + ["Salario_Medio_R$"]].copy()

                final_df = pd.concat([final_df, extras], ignore_index=True)


                # Garante que haja 30 cidades

                if len(final_df) < top_n:
                    logger.warning(f"Apenas {len(final_df)} cidades foram obtidas. Preenchendo com cidades fictícias.")
                    for i in range(top_n - len(final_df)):
                        final_df.loc[len(final_df)] = {
                            "Municipio": f"CidadeFicticia{i+1}",
                            "Populacao": 0,
                            "PIB": 0,
                            "Empresas_Segmento": 0,
                            "Empresas_Perfil_Canal": 0,
                            "Salario_Medio_R$": 0
                        }

        return final_df.head(top_n)

    except Exception as e:
        logger.error(f"Erro ao processar segmentos '{segmentos}': {e}")
        return pd.DataFrame(columns=["Municipio", "Populacao", "PIB", "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"])

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
