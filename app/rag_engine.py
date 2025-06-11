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

STOPWORDS = {"para", "com", "sem", "de", "e", "ou", "por", "em", "da", "do", "no", "na", "das", "dos"}

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

logger.warning(f"[DEBUG] Abas lidas do Excel: {sheet_names}")

for nome, df in sheets_dict.items():
    logger.warning(f"[DEBUG] {nome}: colunas = {df.columns.tolist()}")

sheets_dict = pd.read_excel("Tabela 14.xlsx", sheet_name=None)
sheet_names = list(sheets_dict.keys())


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


equivalencias_semanticas_canais = {
    
    # Segmentos de canais desejados
    "tecnologia": ["ti", "software", "hardware", "sistemas", "desenvolvimento de software", "startups"],
    "gestão": ["administração", "gerenciamento", "gestão empresarial", "consultoria em gestão"],
    "informática": ["tecnologia da informação", "infraestrutura de ti", "serviços de informática", "manutenção de computadores"],
    "internet": ["web", "rede", "provedor de internet", "plataformas digitais", "aplicações online", "serviços online"],
    "consultoria": ["assessoria", "consultoria empresarial", "consultoria estratégica", "serviços especializados"]
}


def expandir_termos_por_equivalencia(lista_termos: list[str], base: dict) -> set:
    termos_expandidos = set()
    for termo in lista_termos:
        termo = termo.lower()
        similares = base.get(termo, [])
        termos_expandidos.update([termo] + similares)
    return termos_expandidos


def extrair_dados_segmentos_cliente_e_canais(segmentos_cliente: list[str], top_n: int = 30):

    #   Extrai informações das cidades com base nos segmentos informados pelo cliente (Empresa) 
    #   e nos segmentos fixos definidos para canais de vendas.

    #etorna:
    #   DataFrame com as colunas:
    #   - Municipio
    #   - Empresas_Segmento (segmentos da empresa)
    #   - Empresas_Perfil_Canal (segmentos fixos de canais)


    termos_cliente = set()

    for termo in segmentos_cliente:
        termos_cliente.update(normalizar_segmentos_inteligente(termo, descricoes_cnae, embeddings_cnae))


    segmentos_canais_input = list(equivalencias_semanticas_canais.keys())

    termos_canais = set()
    
    for termo in segmentos_canais_input:
        normalizados = normalizar_segmentos_inteligente(termo, descricoes_cnae, embeddings_cnae)   
        termos_canais.update(normalizados)
    
    resultados = {}

    for nome_aba in sheet_names:
        df = sheets_dict[nome_aba].copy()

        if COLUNA_ATIVIDADE not in df.columns:
            continue

        df.rename(columns={"Seções e divisões da classificação de atividades": COLUNA_ATIVIDADE}, inplace=True)

        df.columns = [
            "Municipio", "Codigo_CNAE", COLUNA_ATIVIDADE,
            "Unidades_Locais", "Pessoal_Total", "Pessoal_Assalariado", "Assalariado_Medio",
            "Remuneracao_Mil_R$", "Salario_Medio_SM", "Salario_Medio_R$"
        ]

        df = df[df["Municipio"].notna()]
        df = df[~df["Municipio"].astype(str).str.contains("Munic|Tabela|Total", na=False)]
        df = df[~df["Unidades_Locais"].astype(str).isin(["-", "nan"])]
        df["Unidades_Locais"] = pd.to_numeric(df["Unidades_Locais"], errors="coerce")
        df = df[df["Unidades_Locais"].notna()]

        df[COLUNA_ATIVIDADE] = df[COLUNA_ATIVIDADE].astype(str).str.lower()

    contagem_segmento = contar_empresas_por_segmento(df, termos_cliente)
    contagem_canal = contar_empresas_por_segmento(df, termos_canais)

    for cidade in set(contagem_segmento.keys()).union(contagem_canal.keys()):
        if cidade not in resultados:
            resultados[cidade] = {"Empresas_Segmento": 0, "Empresas_Perfil_Canal": 0}

        resultados[cidade]["Empresas_Segmento"] += contagem_segmento.get(cidade, 0)
        resultados[cidade]["Empresas_Perfil_Canal"] += contagem_canal.get(cidade, 0)

    df_final = pd.DataFrame([
        {"Municipio": k, "Empresas_Segmento": v["Empresas_Segmento"], "Empresas_Perfil_Canal": v["Empresas_Perfil_Canal"]}
        for k, v in resultados.items()

        ])

    logger.warning(f"[DEBUG] Linhas encontradas: {len(df_final)}")
    logger.warning(f"[DEBUG] Amostra do DataFrame:\n{df_final.head()}")

    return df_final.sort_values(by="Empresas_Segmento", ascending=False).head(top_n).reset_index(drop=True)


def buscar_similares_embedding(termo: str, descricoes: list[str], threshold: float = 0.35, top_n: int = 3) -> list[str]:
    """
    Retorna as descrições mais semelhantes ao termo fornecido com base em embeddings,
    fuzzy matching e fallback para match parcial por string.
    """
    try:
        termo_emb = get_embedding(termo)
        if not termo_emb:
            logger.warning(f"Embedding nulo para termo: {termo}")
            return []

        # Calcula similaridade com todas as descrições
        pontuacoes = []
        for descricao in descricoes:
            desc_emb = get_embedding(descricao)
            if desc_emb:
                score = cosine_similarity(termo_emb, desc_emb)
                pontuacoes.append((descricao, score))

        # Filtra por threshold
        filtradas = [desc for desc, score in pontuacoes if score >= threshold]
        if filtradas:
            return sorted(filtradas, key=lambda x: -dict(pontuacoes)[x])[:top_n]

        # Fallback: fuzzy matching
        fuzzy = get_close_matches(termo, descricoes, n=top_n, cutoff=0.5)
        if fuzzy:
            logger.info(f"Fuzzy matching usado para termo '{termo}'")
            return fuzzy

        # Último recurso: match parcial
        parciais = [desc for desc in descricoes if termo.lower() in desc.lower()]
        if parciais:
            return parciais[:top_n]

        return []

    except Exception as e:
        logger.error(f"Erro em buscar_similares_embedding: {e}")
        return []


def normalizar_segmentos_inteligente(termo_usuario: str, descricoes_cnae: list[str], embeddings_cnae: list[list[float]], usar_openai_fallback: bool = False) -> list[str]:
    """
    Normaliza um termo informado com base nas descrições CNAE e retorna os termos mais semelhantes.
    Usa: equivalência semântica → embedding → fuzzy → fallback
    """
    termo = termo_usuario.strip().lower()

    # 1. Verifica se existe correspondência direta no dicionário de equivalências

    for chave, lista in equivalencias_semanticas.items():
        if termo == chave or termo in lista:
            return list(set([chave] + lista))

    # 2. Similaridade por embedding
    try:
        emb_termo = get_embedding(termo)
        if not emb_termo:
            logger.warning(f"Embedding não encontrado para termo: {termo}")
        else:
            similaridades = [cosine_similarity(emb_termo, emb) for emb in embeddings_cnae]
            top_indices = np.argsort(similaridades)[::-1]
            top_descricoes = [descricoes_cnae[i] for i in top_indices if similaridades[i] > 0.35][:5]
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
            prompt = f"""Você é um classificador inteligente. A seguir, está um termo de atividade empresarial: '{termo_usuario}'.
Retorne uma descrição de atividade CNAE brasileira mais próxima possível.
"""
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


def buscar_cidades_na_openai(segmentos: list[str], cidades_existentes: list[str], faltantes: int):
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
        import io
        tabela_csv = resposta.choices[0].message.content.strip()

        try:
            df = pd.read_csv(io.StringIO(tabela_csv))
            return df  # ✅ Retorna apenas se o CSV foi lido com sucesso
        except Exception as e:
            logger.error(f"Erro ao ler CSV gerado pela OpenAI: {e}\nConteúdo recebido:\n{tabela_csv}")
            return pd.DataFrame()  # Só retorna vazio se deu erro
        
    except Exception as e:
        logger.error(f"Erro ao buscar cidades com OpenAI: {e}")
        return pd.DataFrame()


def filtrar_municipios_por_segmentos_multiplos(segmentos_textuais: str, top_n: int = 30) -> pd.DataFrame:
    
    """
    Função intermediária que transforma texto solto em lista de segmentos e
    executa a busca por municípios com base no novo motor inteligente.
    """

    logger.warning(f"[DEBUG] Segmentos processados para busca: {segmentos_lista}")

    
    # Converte entrada textual em lista de termos (ex: "agro, logística")
    segmentos_lista = [
        seg.strip().lower()
        for seg in segmentos_textuais.replace(",", " ").split()
        if seg.strip().lower() not in STOPWORDS and len(seg.strip()) > 2
    ]

    if not segmentos_lista:
        logger.warning("Nenhum segmento válido informado.")
        return pd.DataFrame(columns=["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"])

    # Aponta diretamente para o motor novo e unificado
    df_resultado = extrair_dados_segmentos_cliente_e_canais(segmentos_lista, top_n=top_n)

    if df_resultado.empty:
        logger.warning("Nenhum município encontrado para os segmentos informados.")

    return df_resultado


def gerar_tabela_html(dataframe: pd.DataFrame) -> str:
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

# Exportáveis globais
try:
    descricoes_cnae = set()
    for nome_aba in sheet_names:
        df_sheet = sheets_dict[nome_aba]
        if COLUNA_ATIVIDADE in df_sheet.columns:
            descricoes = df_sheet[COLUNA_ATIVIDADE].dropna().unique().tolist()
            descricoes_cnae.update(descricoes)
    descricoes_cnae = list(descricoes_cnae)
    embeddings_cnae = [get_embedding(desc) for desc in descricoes_cnae]
except Exception as e:
    descricoes_cnae = []
    embeddings_cnae = []
    logger.warning(f"Erro ao preparar descrições e embeddings globais: {e}")

import re

def cnae_bate_com_qualquer_termo(cnae: str, termos: set[str]) -> bool:
    tokens = set(re.findall(r"\w+", cnae.lower()))
    return any(t in tokens for t in termos)

def contar_empresas_por_segmento(df: pd.DataFrame, termos: set[str], coluna_atividade: str = COLUNA_ATIVIDADE) -> dict[str, int]:
    contagem = {}
    for _, row in df.iterrows():
        cidade = row["Municipio"]
        atividade = str(row[coluna_atividade]).lower()
        unidades = row["Unidades_Locais"]

        if cidade not in contagem:
            contagem[cidade] = 0

        if cnae_bate_com_qualquer_termo(atividade, termos):
            contagem[cidade] += unidades

    return contagem
