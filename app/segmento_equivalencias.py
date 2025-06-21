import unicodedata
from typing import List, Set
import pandas as pd
import re


# Equivalências semânticas para segmentos comuns no Brasil (indústria, construção, saúde etc.)



# Equivalências semânticas para segmentos comuns no Brasil (indústria, construção, saúde etc.)

EQUIVALENCIAS_SEGMENTOS = {
    "indústria": [

        # Todos os subtipos industriais
        "indústria alimentícia", "indústria de alimentos", "fábrica de alimentos", "processamento de alimentos", "indústria de bebidas", "fábrica de refrigerantes", "fábrica de laticínios", "frigorífico",
        "indústria química", "fábrica de produtos químicos", "indústria de fertilizantes", "indústria de defensivos", "fábrica de tintas", "indústria farmacêutica", "laboratório químico",
        "indústria metalúrgica", "metalúrgica", "fábrica de estruturas metálicas", "fundição", "siderúrgica", "fábrica de peças metálicas", "usinagem",
        "indústria têxtil", "tecelagem", "fiação", "confecção de roupas", "indústria de vestuário", "malharia", "fábrica de tecidos",
        "indústria de calçados", "fábrica de calçados", "calçadista", "indústria de acessórios",
        "indústria moveleira", "fábrica de móveis", "marcenaria", "indústria de móveis planejados", "fábrica de colchões",
        "indústria de papel e celulose", "fábrica de papel", "indústria de celulose", "indústria gráfica",
        "indústria de plástico", "fábrica de plásticos", "transformação de plásticos", "indústria de embalagens plásticas",
        "indústria de borracha", "fábrica de borracha", "indústria de pneus", "indústria de artefatos de borracha",
        "indústria de produtos minerais", "indústria de cerâmica", "indústria de vidro", "fábrica de cimento", "fábrica de blocos",
        "indústria automotiva", "fábrica de veículos", "autopeças", "fábrica de motores", "montadora de automóveis",
        "indústria eletrônica", "fábrica de eletrônicos", "indústria de componentes eletrônicos", "montadora de eletroeletrônicos",
        "indústria de informática", "fábrica de computadores", "montadora de hardware", "indústria de equipamentos de informática",
        "indústria de máquinas e equipamentos", "fábrica de máquinas", "indústria de equipamentos industriais", "fábrica de ferramentas",
        "indústria naval", "indústria de embarcações", "estaleiro", "fábrica de barcos",
        "indústria aeroespacial", "indústria aeronáutica", "fábrica de aviões", "manutenção aeronáutica",
        "indústria de mobiliário urbano", "fábrica de abrigos", "fábrica de postes", "fábrica de lixeiras",
        "indústria de brinquedos", "fábrica de brinquedos", "indústria de jogos", "fábrica de bonecos",
        "indústria de cosméticos", "fábrica de cosméticos", "indústria de perfumaria", "fábrica de produtos de higiene",
        "indústria gráfica", "gráfica", "impressão offset", "editora", "serviços gráficos",
        "indústria de defesa", "indústria bélica", "fábrica de munições", "fábrica de equipamentos militares"
    ],

 # Segmentos de serviço e outros sugeridos

    "serviços": [
        "prestação de serviços", "serviços empresariais", "serviços especializados", "serviços de consultoria", "serviços técnicos", "serviços administrativos", "terceirização", "serviços profissionais", "serviços de apoio", "serviços de manutenção", "serviços gerais", "serviços de limpeza", "serviços de segurança", "serviços de transporte", "serviços de tecnologia", "serviços de informação", "serviços de saúde", "serviços educacionais"
    ],
    "projetos": [
        "projeto técnico", "gerenciamento de projetos", "implantação de projetos", "projeto de engenharia", "projeto de arquitetura", "projeto industrial", "projeto de software", "projeto de construção", "projeto de infraestrutura", "projeto logístico", "consultoria em projetos"
    ],
    "logística": [
        "logística", "logística integrada", "logística reversa", "logística de distribuição", "gestão logística", "transporte e logística", "consultoria logística", "armazenagem", "centro de distribuição", "supply chain", "cadeia de suprimentos"
    ],
    "logistica": [],
    "transportes": [
        "transporte rodoviário", "transporte ferroviário", "transporte aéreo", "transporte marítimo", "transporte fluvial", "empresa de transporte", "logística de transportes", "serviço de frete", "carga e descarga", "transporte de passageiros", "transporte de cargas", "empresa de ônibus", "empresa de caminhão"
    ],
    "transformação": [
        "indústria de transformação", "transformação de plásticos", "transformação de metais", "transformação de madeira", "transformação de alimentos", "processamento industrial", "fabricação", "industrialização"
    ],
    "transformacao": [],
    "comércio": [
        "comércio atacadista", "comércio varejista", "loja", "supermercado", "minimercado", "distribuidora", "comércio de alimentos", "comércio eletrônico", "e-commerce", "varejo", "atacado", "centro comercial", "shopping"
    ],
    "comercio": [],
    "varejo": [
        "comércio varejista", "loja de varejo", "varejista", "supermercado", "loja física", "loja online"
    ],
    "agro": [
        "agropecuária", "agronegócio", "agrícola", "empresa agrícola", "fazenda", "produtor rural", "produção agrícola", "produção agropecuária", "agricultura", "pecuária", "cooperativa agrícola", "agroindústria"
    ],
    "agronegócio": [
        "agro", "agropecuária", "agrícola", "agribusiness", "empresa rural", "empresa agrícola", "fazenda", "produtor rural", "produção agrícola", "produção agropecuária", "agricultura", "pecuária", "cooperativa agrícola", "agroindústria"
    ],
    "agronegocio": [],
    "manufatura": [
        "fabricação", "produção industrial", "indústria de transformação", "montagem industrial", "processamento industrial", "linha de produção", "usina", "fábrica", "transformação"
    ],
    "tecnologia": [
        "tecnologia da informação", "TI", "informática", "automação", "software", "hardware", "sistemas de informação", "consultoria em TI", "empresa de tecnologia", "empresa de software", "desenvolvimento de sistemas", "infraestrutura de TI"
    ],
    "ti": [
        "tecnologia da informação", "tecnologia", "informática", "automação", "software", "hardware", "sistemas de informação"
    ],
    "educação": [
        "ensino", "escola", "universidade", "instituto", "curso", "educacional", "faculdade", "treinamento", "capacitação", "educação profissional", "ensino técnico"
    ],
    "educacao": [],
    "construção": [
        "construção civil", "obra", "empreiteira", "construtora", "engenharia civil", "reforma", "manutenção predial", "projetos de construção"
    ],
    "construcao": [],
    "saúde": [
        "clínica", "hospital", "laboratório", "consultório", "odontologia", "empresa de saúde", "assistência médica", "plano de saúde", "medicina", "enfermagem", "fisioterapia", "terapia", "nutrição", "psicologia"
    ],
    "saude": [],
    "energia": [
        "energia elétrica", "energia renovável", "energia solar", "energia eólica", "usina", "geração de energia", "distribuição de energia", "concessionária de energia"
    ],
    "financeiro": [
        "finanças", "banco", "instituição financeira", "contabilidade", "consultoria financeira", "investimentos", "cooperativa de crédito", "seguradora", "corretora"
    ],
    "serviço público": [
        "empresa pública", "autarquia", "órgão público", "prefeitura", "governo", "secretaria", "serviço municipal", "serviço estadual", "serviço federal"
    ],
    "servico publico": [],

    # Subsegmentos industriais

    "indústria alimentícia": [
        "indústria de alimentos", "fábrica de alimentos", "processamento de alimentos", "indústria de bebidas", "fábrica de refrigerantes", "fábrica de laticínios", "frigorífico"
    ],
    "indústria química": [
        "fábrica de produtos químicos", "indústria de fertilizantes", "indústria de defensivos", "fábrica de tintas", "indústria farmacêutica", "laboratório químico"
    ],
    "indústria metalúrgica": [
        "metalúrgica", "fábrica de estruturas metálicas", "fundição", "siderúrgica", "fábrica de peças metálicas", "usinagem"
    ],
    "indústria têxtil": [
        "tecelagem", "fiação", "confecção de roupas", "indústria de vestuário", "malharia", "fábrica de tecidos"
    ],
    "indústria de calçados": [
        "fábrica de calçados", "calçadista", "indústria de acessórios"
    ],
    "indústria moveleira": [
        "fábrica de móveis", "marcenaria", "indústria de móveis planejados", "fábrica de colchões"
    ],
    "indústria de papel e celulose": [
        "fábrica de papel", "indústria de celulose", "indústria gráfica"
    ],
    "indústria de plástico": [
        "fábrica de plásticos", "transformação de plásticos", "indústria de embalagens plásticas"
    ],
    "indústria de borracha": [
        "fábrica de borracha", "indústria de pneus", "indústria de artefatos de borracha"
    ],
    "indústria de produtos minerais": [
        "indústria de cerâmica", "indústria de vidro", "fábrica de cimento", "fábrica de blocos"
    ],
    "indústria automotiva": [
        "fábrica de veículos", "autopeças", "fábrica de motores", "montadora de automóveis"
    ],
    "indústria eletrônica": [
        "fábrica de eletrônicos", "indústria de componentes eletrônicos", "montadora de eletroeletrônicos"
    ],
    "indústria de informática": [
        "fábrica de computadores", "montadora de hardware", "indústria de equipamentos de informática"
    ],
    "indústria de máquinas e equipamentos": [
        "fábrica de máquinas", "indústria de equipamentos industriais", "fábrica de ferramentas"
    ],
    "indústria naval": [
        "indústria de embarcações", "estaleiro", "fábrica de barcos"
    ],
    "indústria aeroespacial": [
        "indústria aeronáutica", "fábrica de aviões", "manutenção aeronáutica"
    ],
    "indústria de mobiliário urbano": [
        "fábrica de abrigos", "fábrica de postes", "fábrica de lixeiras"
    ],
    "indústria de brinquedos": [
        "fábrica de brinquedos", "indústria de jogos", "fábrica de bonecos"
    ],
    "indústria de cosméticos": [
        "fábrica de cosméticos", "indústria de perfumaria", "fábrica de produtos de higiene"
    ],
    "indústria gráfica": [
        "gráfica", "impressão offset", "editora", "serviços gráficos"
    ],
    "indústria de defesa": [
        "indústria bélica", "fábrica de munições", "fábrica de equipamentos militares"
    ],

    # Construção Civil (detalhada)

    "construção civil": [
        "construção de edifícios", "obras de infraestrutura", "engenharia civil", "serviços especializados para construção", "alvenaria", "empreiteira", "construtora", "construção residencial", "construção não-residencial", "obras de urbanização", "loteamento", "serviços de terraplenagem", "demolição", "pintura predial", "reforma de imóveis"
    ],
    "construcao civil": [],  # será preenchido para equivalência sem acento
    "materiais de construção": [
        "loja de materiais de construção", "distribuidora de materiais de construção", "fábrica de blocos", "fábrica de cimento", "madeireira", "depósito de material de construção"
    ],
    "engenharia elétrica": [
        "instalações elétricas", "projetos elétricos", "manutenção elétrica"
    ],
    "engenharia hidráulica": [
        "instalações hidráulicas", "projetos hidráulicos", "manutenção hidráulica"
    ],
    "infraestrutura viária": [
        "construção de estradas", "pavimentação", "obras viárias", "duplicação de rodovias"
    ],
    "fundação e sondagem": [
        "empresa de sondagem", "fundação de obras", "ensaios de solo"
    ],
    "arquitetura e urbanismo": [
        "escritório de arquitetura", "projetos arquitetônicos", "urbanismo"
    ],
    "paisagismo": [
        "empresa de paisagismo", "projetos paisagísticos", "jardinagem"
    ],
    "revestimentos e acabamentos": [
        "empresa de pintura", "empresa de revestimento", "forro e gesso", "revestimento cerâmico"
    ],
    "manutenção predial": [
        "limpeza pós-obra", "conservação predial", "manutenção de elevadores"
    ],

    # Saúde (detalhada)

    "saúde": [
        "atividades hospitalares", "clínicas médicas", "laboratórios de análises clínicas", "consultórios médicos", "hospitais", "planos de saúde", "consultório odontológico", "centro cirúrgico", "pronto atendimento", "ambulatório"
    ],
    "saude": [],  # sem acento
    "clínicas especializadas": [
        "clínica de fisioterapia", "clínica de ortopedia", "clínica de oftalmologia", "clínica de cardiologia", "clínica de dermatologia", "clínica psiquiátrica", "clínica de ginecologia", "clínica de oncologia"
    ],
    "clinicas especializadas": [],
    "laboratório": [
        "laboratório de análises clínicas", "laboratório de biomedicina", "laboratório de genética", "laboratório de patologia"
    ],
    "laboratorio": [],
    "farmacêutico": [
        "farmácia", "distribuidora de medicamentos", "indústria farmacêutica", "drogaria", "manipulação"
    ],
    "farmaceutico": [],
    "diagnóstico por imagem": [
        "clínica de radiologia", "diagnóstico por imagem", "tomografia", "ressonância magnética"
    ],
    "diagnostico por imagem": [],
    "home care": [
        "atendimento domiciliar", "empresa de home care", "cuidados domiciliares"
    ],
    "plano de saúde": [
        "operadora de saúde", "seguro saúde", "planos odontológicos"
    ],
    "plano de saude": [],
    "serviços de enfermagem": [
        "empresa de enfermagem", "cooperativa de enfermagem", "home care de enfermagem"
    ],
    "servicos de enfermagem": [],
    "assistência hospitalar": [
        "pronto-socorro", "UPA", "unidade de pronto atendimento", "emergência médica"
    ],
    "assistencia hospitalar": [],
    "reabilitação": [
        "clínica de reabilitação", "centro de reabilitação", "fisioterapia", "fonoaudiologia", "terapia ocupacional"
    ],
    "reabilitacao": [],
}

# Preencher equivalências sem acento de forma DRY

def remover_acentos(txt):
    return unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode("utf-8").lower()

for chave in list(EQUIVALENCIAS_SEGMENTOS.keys()):
    chave_sem_acentos = remover_acentos(chave)
    if chave_sem_acentos not in EQUIVALENCIAS_SEGMENTOS or not EQUIVALENCIAS_SEGMENTOS[chave_sem_acentos]:
        EQUIVALENCIAS_SEGMENTOS[chave_sem_acentos] = EQUIVALENCIAS_SEGMENTOS[chave]


def normalizar_termo_segmento(termo_usuario: str) -> List[str]:
    """
    Dado um termo (ex: 'industria', 'construção civil', 'saúde', etc), retorna todos equivalentes.
    Sempre busca sem acento e minúsculo.
    """
    termo = remover_acentos(termo_usuario.strip().lower())
    equivalentes = EQUIVALENCIAS_SEGMENTOS.get(termo)
    if equivalentes:
        return [termo_usuario] + equivalentes
    return [termo_usuario]


# Função principal para expandir termos

 
def expandir_equivalencias_lista(termos_usuario: List[str]) -> List[str]:
    """
    Dada uma lista de termos do usuário, retorna uma lista expandida de equivalências (sem repetições).
    """
    resultado: Set[str] = set()
    for termo in termos_usuario:
        equivalentes = normalizar_termo_segmento(termo)
        resultado.update([remover_acentos(t.strip()) for t in equivalentes])
    return list(resultado)

def get_equivalentes_segmento_cliente(segmentos_cliente):
    return expandir_equivalencias_lista(segmentos_cliente)

def contar_empresas_por_segmento(df, col_segmento, col_cidade, termos_segmento):
    """
    Conta empresas por cidade para os termos de segmento informados.
    """
    df_filtrado = buscar_segmentos_em_df(df, [col_segmento], termos_segmento)
    return df_filtrado[col_cidade].value_counts()

def top_n_cidades(series_cidade, n=30):
    return series_cidade.head(n).index.tolist()

def get_equivalentes_ti():
    return expandir_equivalencias_lista(["tecnologia", "ti"])

def contar_empresas_tecnologia_nessas_cidades(df, col_segmento, col_cidade, cidades_top30):
    termos_ti = get_equivalentes_ti()
    df_ti = buscar_segmentos_em_df(df, [col_segmento], termos_ti)
    filtro_cidades = df_ti[col_cidade].isin(cidades_top30)
    return df_ti[filtro_cidades][col_cidade].value_counts()

def gerar_tabela_clientes_tecnologia():
    df = carregar_dados_df()
    col_segmento = "Nome do CNAE"  # ou o nome correto da coluna de segmento
    col_cidade = "Município"       # ou o nome correto da coluna de cidade

    # 1. Segmentos do cliente (exemplo: pode vir de input, aqui fixo)
    segmentos_cliente = ["agronegócio", "comércio"]

    # 2. Equivalentes do cliente
    termos_cliente = get_equivalentes_segmento_cliente(segmentos_cliente)

    # 3. Conta empresas por cidade
    series_cidade = contar_empresas_por_segmento(df, col_segmento, col_cidade, termos_cliente)

    # 4. Top 30 cidades
    cidades_top30 = top_n_cidades(series_cidade, 30)

    # 5. Conta empresas de TI/tecnologia nessas cidades
    series_ti = contar_empresas_tecnologia_nessas_cidades(df, col_segmento, col_cidade, cidades_top30)

    # 6. Gera tabela final
    result = []
    for cidade in cidades_top30:
        total_cliente = series_cidade.get(cidade, 0)
        total_ti = series_ti.get(cidade, 0)
        result.append({"Cidade": cidade, "Empresas Segmento Cliente": total_cliente, "Empresas Tecnologia/TI": total_ti})
    df_result = pd.DataFrame(result)
    return df_result



def buscar_segmentos_em_df(df: pd.DataFrame, colunas_busca: List[str], termos_usuario: List[str]) -> pd.DataFrame:
    """
    Busca linhas no DataFrame onde qualquer equivalente dos segmentos do usuário aparece nas colunas indicadas.
    - df: DataFrame pandas (ex: planilha CNAE)
    - colunas_busca: lista de nomes de colunas para procurar (ex: ['Descrição'])
    - termos_usuario: lista de segmentos informados pelo usuário
    Retorna um DataFrame filtrado.
    """
    
    equivalentes = expandir_equivalencias_lista(termos_usuario)
    pattern = '|'.join(map(re.escape, equivalentes))
    mask_total = None
    for coluna in colunas_busca:
        if coluna not in df.columns:
            print(f"⚠️ Atenção: coluna '{coluna}' não encontrada no DataFrame.")
            continue
        col_normalizada = df[coluna].astype(str).apply(remover_acentos)
        mask = col_normalizada.str.contains(pattern, case=False, na=False)
        mask_total = mask if mask_total is None else (mask_total | mask)
    return df[mask_total] if mask_total is not None else df.iloc[0:0]

# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo de termos digitados pelo usuário:
    termos = ["Serviços", "Logística", "Agronegócio", "Projetos", "Transformação", "TI", "Comércio", "Varejo", "Educação", "Energia", "Saúde", "Construção", "Manufatura", "Financeiro", "Serviço Público"]
    todos_equivalentes = expandir_equivalencias_lista(termos)
    print("Equivalências expandidas:", todos_equivalentes)

    # Exemplo de uso em DataFrame:
    df = pd.read_excel("Tabela 14.xlsx")
    filtrado = buscar_segmentos_em_df(df, ["Descrição"], termos)
    print(filtrado)

def carregar_dados_df():
    """
    Carrega todos os dados das abas do Excel (Tabela_CNAE.xlsx) em um único DataFrame,
    adicionando uma coluna 'Aba' para origem.
    """
    df_dict = pd.read_excel("Tabela 14.xlsx", sheet_name=None)
    dfs = []
    for aba, df in df_dict.items():
        if not df.empty:
            df['Aba'] = aba
            dfs.append(df)
    df_total = pd.concat(dfs, ignore_index=True)
    return df_total

