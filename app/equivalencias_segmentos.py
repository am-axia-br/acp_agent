import unicodedata

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
    "industria": [],  # será preenchido no final para equivalência sem acento

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

# Função principal para expandir termos
def normalizar_termo_segmento(termo_usuario):
    """
    Dado um termo (ex: 'industria', 'construção civil', 'saúde', etc), retorna todos equivalentes.
    Sempre busca sem acento e minúsculo.
    """
    termo = remover_acentos(termo_usuario.strip().lower())
    equivalentes = EQUIVALENCIAS_SEGMENTOS.get(termo)
    if equivalentes:
        return [termo_usuario] + equivalentes
    return [termo_usuario]