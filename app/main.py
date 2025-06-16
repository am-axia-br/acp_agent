from log_config import get_logger
logger = get_logger(__name__)

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import pandas as pd
import os
import traceback
import json
from dotenv import load_dotenv
from mail import enviar_email
from openai import OpenAI

STOPWORDS = {"para", "com", "sem", "de", "e", "ou", "por", "em", "da", "do", "no", "na", "das", "dos"}

from rag_engine import (
    filtrar_municipios_por_segmentos_multiplos as filtrar_municipios_por_segmento,
    gerar_tabela_html,
    normalizar_segmentos_inteligente,
    descricoes_cnae,
    embeddings_cnae,
    buscar_cidades_na_openai
)

from rag_parcerias import buscar_conhecimento

def consultar_taxa_conversao_openai(segmentos):
    try:
        prompt = f"""Para empresas do segmento {', '.join(segmentos)}, qual é a taxa média de conversão de:
- Prospecção para oportunidade?
- Oportunidade para venda?
Dê um número percentual para cada uma, no Brasil, em mercados B2B."""

        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor especialista em vendas B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        texto = resposta.choices[0].message.content.strip()
        matches = re.findall(r"(\d{1,2})%", texto)
        if len(matches) >= 2:
            p_para_o = int(matches[0]) / 100
            o_para_v = int(matches[1]) / 100
            return p_para_o * o_para_v  # taxa final composta
        else:
            logger.warning("Taxa de conversão não encontrada, usando padrão 20%")
            return 0.2
    except Exception as e:
        logger.error(f"Erro ao consultar taxa de conversao: {str(e)}")
        return 0.2


# DEFINA A FUNÇÃO ANTES DE USÁ-LA
def novo_diagnostico():
    return {
        "nome": None,
        "empresa": None,
        "whatsapp": None,
        "email": None,
        "diagnostico": [],
        "etapa_atual": 0,
        "iniciado": False,
        "municipios": [],
        "resumo": "",
        "mensagem_usuario": "",
        "resposta_ia": ""
    }

# VARIÁVEL GLOBAL
data = novo_diagnostico()



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Aplicacao FastAPI iniciada")

@app.on_event("startup")
def indexar_automaticamente():
    try:
        from rag_parcerias import indexar_documentos
        total = indexar_documentos()
        logger.info(f"Indexacao automatica no startup concluida com {total} arquivos.")
    except Exception as e:
        logger.warning(f"Indexacao automatica ignorada: {e}")



@app.get("/")
def home():
    global data
    data = novo_diagnostico()
    logger.info("Pagina inicial acessada e dados resetados")
    return FileResponse("static/index.html")

@app.get("/chat")

def chat_get():
    return {"erro": "Metodo GET nao permitido nesta rota. Use POST com corpo JSON contendo 'mensagem'."}

class Mensagem(BaseModel):
    mensagem: str

perguntas = [
    "Qual o site da empresa?",
    "Quais segmentos a empresa atende?",
    "Poderia citar 3 clientes atuais?",
    "Quais dores a sua empresa resolve e quais beneficios sao levados aos clientes?",
    "Quais os seus diferenciais?",
    "Quais os produtos e servicos vendidos?",
    "Qual o modelo de negocio da empresa? Comercializa licencas? Cobra mensalidade? Cobra projeto?",
    "Qual o ticket medio dos negocios?",
    "Qual o ciclo medio de vendas (em dias)?",
    "Qual a sua expectativa de vendas de novos clientes pelos canais mensalmente?"
]

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    msg = body.get("mensagem", "").strip()
    logger.info(f"Mensagem recebida: {msg}")

    if not data["iniciado"]:
        data["iniciado"] = True
        return {"mensagem": "Ola... Para comecarmos o diagnostico, me fale de onde você fala..."}

    if not data.get("origem"):
        data["origem"] = msg
        return {"pergunta": "Qual o seu nome?"}

    if not data["nome"]:
        data["nome"] = msg
        return {"pergunta": "Qual o nome da sua empresa?"}

    if not data["empresa"]:
        data["empresa"] = msg
        return {"pergunta": "Qual o seu WhatsApp? (Ex: DDD9XXXXYYYY)"}

    if not data["whatsapp"]:
        msg = re.sub(r"\D", "", msg)
        if re.match(r"^\d{11}$", msg):
            data["whatsapp"] = msg
            return {"pergunta": "Qual o seu e-mail?"}
        logger.warning("WhatsApp em formato invalido")
        return {"pergunta": "Formato invalido. Exemplo: DDD9XXXXYYYY"}

    if not data["email"]:
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,4}$", msg):
            data["email"] = msg
            return {"mensagem": "Obrigado! Agora vamos entender melhor sua empresa.", "pergunta": perguntas[0]}
        logger.warning("E-mail invalido")
        return {"pergunta": "E-mail invalido. Envie um formato valido."}

    if data["etapa_atual"] < len(perguntas):
        if data["etapa_atual"] in [7, 8, 9]:
            valor = re.sub(r"[^\d]", "", msg)
            if not valor.isdigit():
                return {"pergunta": "Por favor, responda apenas com numeros (sem R$, pontos ou virgulas)."}
            msg = valor

        data["diagnostico"].append(msg)
        logger.info(f"Respostas coletadas até agora: {len(data['diagnostico'])}")

        data["etapa_atual"] += 1
        if data["etapa_atual"] < len(perguntas):
            return {"pergunta": perguntas[data["etapa_atual"]]}
        else:
            if len(data["diagnostico"]) < len(perguntas):
                logger.error("Diagnóstico incompleto. Esperado respostas.")
                return {"mensagem": "Erro: Respostas incompletas. Por favor, reinicie o diagnóstico.", "resumo": "Erro: respostas insuficientes.", "email": data["email"]}
            try:
                prompt = gerar_prompt(data)
                data["prompt"] = prompt
                data["finalizado"] = True
                logger.info("Prompt preparado com sucesso")
                return {
                    "mensagem": "Analisando suas respostas e preparando o diagnóstico...",
                    "loading": True,
                    "email": data["email"]
                }
            except Exception as e:
                print(f"Erro ao preparar prompt: {e}")
                return {
                    "resumo": f"Erro técnico: {str(e)}"
                }

    return {"mensagem": "Diagnostico ja concluido."}

@app.post("/gerar-diagnostico")
async def gerar_diagnostico():
    try:
        resposta = chamar_llm(data["prompt"])
        enviar_email(data, resposta, copia_para=["alexandre.maia@acp.tec.br"])
        logger.info("Diagnostico gerado com sucesso pela LLM e e-mail enviado")
        return {"mensagem": "Diagnostico finalizado! Aqui esta nossa analise baseada nas suas respostas:", "resumo": resposta, "email": data["email"]}
    except Exception as e:
        logger.error(f"Erro ao gerar diagnostico: {str(e)}")
        return {"mensagem": "Ocorreu um erro ao gerar o diagnostico.", "resumo": f"Erro ao gerar sugestao: {str(e)}\n\n{traceback.format_exc()}", "email": data["email"]}

@app.post("/reset")
async def resetar_diagnostico():
    global data
    data = novo_diagnostico()
    return {"status": "resetado"}


@app.post("/reindexar-rag")
async def reindexar_rag():
    try:
        from rag_parcerias import indexar_documentos
        total = indexar_documentos()
        logger.info(f"Reindexacao realizada com {total} arquivos")
        return {"status": "ok", "arquivos_indexados": total}
    except Exception as e:
        logger.error(f"Erro na reindexacao do RAG: {str(e)}")
        return {"status": "erro", "mensagem": str(e), "detalhes": traceback.format_exc()}

def truncar_texto(texto, limite=500):
    return texto[:limite] + "..." if len(texto) > limite else texto

def gerar_prompt(data):

    origem = data.get("origem", "")
    nome = data["nome"]
    empresa = data["empresa"]
    respostas = data["diagnostico"]

    if len(respostas) < 2:
        raise ValueError("Respostas insuficientes para gerar o prompt.")

    site = respostas[0]
    segmentos_raw = respostas[1]
    clientes = respostas[2]
    dores = respostas[3]
    diferenciais = respostas[4]
    produtos = respostas[5]
    modelo_negocio = respostas[6]
    ticket_medio_str = respostas[7]
    ciclo_vendas = respostas[8]
    meta_clientes = respostas[9]


    if isinstance(segmentos_raw, list):
        segmentos_raw = " ".join(segmentos_raw)

    segmento_original = segmentos_raw if len(respostas) > 1 else ""

# ✅ Definir segmentos_lista aqui de forma segura

    segmentos_lista = [
        seg.strip().lower()
        for seg in re.split(r"[,\s]+", segmento_original)
        if seg.strip().lower() not in STOPWORDS and len(seg.strip()) > 2
    ]

    segmentos_normalizados = []
    
    for termo in segmentos_lista:
        normalizados = normalizar_segmentos_inteligente(termo, descricoes_cnae, embeddings_cnae)
        segmentos_normalizados.extend(normalizados)

    segmento = truncar_texto(segmentos_raw)
    clientes = truncar_texto(clientes)
    dores = truncar_texto(dores)
    produtos = truncar_texto(produtos)
    modelo = truncar_texto(modelo_negocio)

    resumo_empresa = detalhar_empresa_openai(site, empresa)
    resumo_produto = detalhar_produto_openai(produtos, segmento)
    resumo_clientes = detalhar_clientes_openai(clientes)

    taxa_pros, taxa_vendas = consultar_taxas_software_b2b()

    try:
        novos_clientes = int(meta_clientes)
        ciclo = int(ciclo_vendas)
    except:
        novos_clientes = 1
        ciclo = 90

    oportunidades = int(novos_clientes / taxa_vendas)
    prospeccoes = int(oportunidades / taxa_pros)

    conhecimento_modelos = buscar_conhecimento_complementado(
        f"Quais modelos de parceria são ideais para empresas como a {empresa}, que atuam com {produtos} no segmento {segmento}?"
    )
    conhecimento_perfis = buscar_conhecimento_complementado(
        f"Quais os perfis de empresas parceiras ideais para {empresa}, considerando que seus clientes são como: {clientes}?"
    )
    conhecimento_servicos = buscar_conhecimento_complementado(
        f"Que serviços agregados são relevantes para empresas que vendem {produtos}, com modelo de negócio baseado em {modelo}?"
    )

    # 🔍 CIDADES – RAG + OpenAI até 30, sem fictícias

    if isinstance(segmento_original, list):
        segmento_original = " ".join(segmento_original)

    cidades_df = filtrar_municipios_por_segmento(segmento_original, top_n=30)


    if cidades_df.empty or any(col not in cidades_df.columns for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]):
        logger.warning("DataFrame de cidades incompleto. Criando estrutura padrão.")
        cidades_df = pd.DataFrame({
            "Municipio": ["CidadeDesconhecida"],
            "Empresas_Segmento": [0],
            "Empresas_Perfil_Canal": [0]
    })


    # Proteção contra colunas ausentes no DataFrame
    import logging
    logger = logging.getLogger("main")

    logger.warning(f"[DEBUG] Colunas retornadas: {list(cidades_df.columns)}")

    for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
        if col not in cidades_df.columns:
            logger.warning(f"Coluna ausente: {col} - preenchendo com valor padrão.")
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    try:
        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"[FALHA] Erro ao ordenar DataFrame por 'Empresas_Segmento': {e}")

    logger.warning(f"[DEBUG] Colunas recebidas: {list(cidades_df.columns)}")

    for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
        if col not in cidades_df.columns:
            logger.warning(f"Coluna ausente: {col} - criando coluna padrão.")
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    try:
        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"[FALHA] Erro ao ordenar por 'Empresas_Segmento': {e}")


    # LOG para debug
    logger.warning(f"[DEBUG] Colunas do DataFrame retornado: {list(cidades_df.columns)}")

    # Garante que colunas essenciais existam antes de qualquer acesso
    colunas_necessarias = ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]
    for col in colunas_necessarias:
        if col not in cidades_df.columns:
            logger.warning(f"Coluna ausente: {col}. Criando com valor padrão.")
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    # Ordena com proteção
    try:
        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Erro ao ordenar DataFrame por 'Empresas_Segmento': {e}")

    logger.warning(f"[DEBUG] Colunas recebidas do DataFrame: {list(cidades_df.columns)}")


    # Previne falhas ao acessar colunas esperadas
    colunas_obrigatorias = ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]
    for col in colunas_obrigatorias:
        if col not in cidades_df.columns:
            logger.warning(f"Coluna ausente: {col}. Criando com valor padrão.")
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"


    import logging
    logger = logging.getLogger("main")

    logger.warning(f"Colunas do DataFrame retornado: {list(cidades_df.columns)}")

    # Corrige ausência das colunas obrigatórias
    for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
        if col not in cidades_df.columns:
            logger.warning(f"Coluna ausente detectada: {col}. Adicionando com valor padrão.")
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    logger.warning(f"Colunas do DataFrame retornado: {list(cidades_df.columns)}")

    # Garante colunas obrigatórias para evitar erro ao ordenar
    
    for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
        if col not in cidades_df.columns:
            logger.warning(f"Coluna ausente detectada: {col}. Adicionando coluna com valor padrão.")
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    # Agora podemos ordenar com segurança

    try:
        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Erro ao ordenar por 'Empresas_Segmento': {e}")


    if len(cidades_df) < 30:

        if cidades_df.empty or "Municipio" not in cidades_df.columns:
            logger.warning("DataFrame de cidades sem coluna 'Municipio'. Criando estrutura padrão.")
            cidades_df = pd.DataFrame({
                "Municipio": ["CidadeDesconhecida"],
                "Empresas_Segmento": [0],
                "Empresas_Perfil_Canal": [0]
            })

        cidades_existentes = cidades_df["Municipio"].tolist()
        faltantes = 30 - len(cidades_df)
        cidades_df_extra = buscar_cidades_na_openai(segmentos_lista, cidades_existentes, faltantes)

        for col in ["Municipio","Empresas_Segmento", "Empresas_Perfil_Canal"]:
            if col not in cidades_df_extra.columns:
                cidades_df_extra[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

        cidades_df = pd.concat([cidades_df, cidades_df_extra], ignore_index=True)
        cidades_df = cidades_df.drop_duplicates(subset="Municipio").reset_index(drop=True)

        faltam = 30 - len(cidades_df)
        if faltam > 0:
            logger.warning(f"Ainda faltam {faltam} cidades. Chamando OpenAI novamente.")
            cidades_df_extra = buscar_cidades_na_openai(segmentos_lista, cidades_df["Municipio"].tolist(), faltam)
            cidades_df = pd.concat([cidades_df, cidades_df_extra], ignore_index=True).drop_duplicates(subset="Municipio").reset_index(drop=True)


        for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
            if col not in cidades_df.columns:
                cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)

    cidades_html = gerar_tabela_html(cidades_df)

    try:
        ticket = float(ticket_medio_str)
        ciclo = int(ciclo_vendas)
        novos_clientes = int(meta_clientes)
    except Exception as e:
        raise ValueError(f"Erro ao converter valores numericos: {str(e)}") from e

    conversao = consultar_taxa_conversao_openai(segmentos_normalizados)

    oportunidades = int(novos_clientes / conversao)
    prospeccoes = int(oportunidades / conversao)

    from datetime import datetime, timedelta
    receita_24_meses = 0
    hoje = datetime.today()

    for canal_index in range(20):
        dias_ate_primeira_venda = 90 + ciclo + (canal_index * 30)
        data_primeira_venda = hoje + timedelta(days=dias_ate_primeira_venda)
        meses_restantes = max(0, 24 - ((data_primeira_venda - hoje).days // 30))
        receita_24_meses += ticket * meses_restantes

    receita_mensal = ticket * novos_clientes * 20

    return f'''
🧠 DIAGNÓSTICO ESTRUTURADO – EMPRESA {empresa.upper()}

🔹 Local informado:
{origem}

🔹 Resumo sobre a empresa:
{resumo_empresa}

🔹 Produtos e Serviços:
{resumo_produto}

🔹 Perfil dos Clientes atendidos:
{resumo_clientes}

🔹 Dores e Benefícios:
{dores}

🔹 Modelo de Negócio:
{modelo_negocio}

🔹 Necessidades de Prospecção:
- Meta Prevista: {novos_clientes} novo cliente/mês  
- Ciclo de Venda: {ciclo} dias  
- Índice de Conversão de Prospecção em Oportunidade: {int(taxa_pros * 100)}%  
- Número de Prospecções: {prospeccoes:,}  
- Índice de Conversão de Oportunidades em Vendas: {int(taxa_vendas * 100)}%  
- Número de Oportunidades: {oportunidades:,}

🔹 Modelos de Parceria:
{conhecimento_modelos}

🔹 Perfis de Empresas Parceiras com FIT:
{conhecimento_perfis}

🔹 Serviços Agregados:
{conhecimento_servicos}

🔹 Cidades com Potencial:
{cidades_html}

🔹 Retorno sobre o Investimento:
Ticket Médio: R${ticket:,.2f}
Receita Mensal com 20 canais: R${receita_mensal:,.2f}
Receita estimada em 24 meses: R${receita_24_meses:,.2f}

🔹 Dicas Estratégicas:
- Inicie com 3 a 5 canais
- Forneça treinamentos e acompanhamento semanal
- Crie campanhas e ações de marketing
- Corrija falhas com feedback dos canais

🔹 Chamada para Ação:
Sua empresa está pronta para crescer com uma estratégia sólida de canais de vendas.
Entre em contato com a AC Partners e comece agora o onboarding comercial com especialistas.
'''

logger.warning(f"[DEBUG FINAL] Total de cidades no resumo: {len(cidades_df)}")
logger.warning(f"[DEBUG FINAL] Amostra:\n{cidades_df.head()}")

def chamar_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor especialista em canais de vendas. Siga exatamente a estrutura em 13 partes numeradas conforme o prompt, sem pular nenhuma parte."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Erro na chamada a API OpenAI")
        raise RuntimeError("Erro na chamada a API OpenAI.") from e

def sugerir_cidades_openai(segmentos, total_necessario=30):
    try:
        prompt = f"Liste {total_necessario} cidades brasileiras com potencial de mercado para empresas que atuam nos segmentos: {', '.join(segmentos)}. Considere demanda, perfil econômico, presença de agronegócio, indústria ou comércio local, dependendo do segmento. Forneça apenas os nomes das cidades, uma por linha."
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um analista de inteligência de mercado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        cidades = resposta.choices[0].message.content.strip().split("\n")
        return [c.strip().split("-")[0].strip() for c in cidades if c.strip()]
    except Exception as e:
        logger.error(f"Erro ao sugerir cidades com OpenAI: {str(e)}")
        return []

def detalhar_empresa_openai(site, nome_empresa):
    try:
        prompt = f"""
Você é um redator técnico. Com base no nome da empresa "{nome_empresa}" e no site "{site}", gere um resumo claro, profissional e objetivo sobre a empresa. Destaque o segmento de atuação, diferenciais e áreas atendidas.
Use no máximo 3 frases.
"""
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um redator corporativo especializado em empresas B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao detalhar empresa: {str(e)}")
        return f"Para mais informações, visite [{nome_empresa}]({site})."

def detalhar_produto_openai(produto_texto, segmento):
    try:
        prompt = f"""
Explique de forma clara e profissional o seguinte produto ou solução voltado ao segmento {segmento}: "{produto_texto}".
Gere 2 frases destacando utilidade e valor estratégico para empresas B2B.
"""
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em posicionamento de soluções B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao detalhar produto: {str(e)}")
        return produto_texto

def detalhar_clientes_openai(clientes_lista):
    try:
        prompt = f"""
Gere um parágrafo curto sobre o perfil dos seguintes clientes atendidos atualmente:
{clientes_lista}
Mostre diversidade, relevância setorial e valor estratégico.
"""
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um analista de portfólio de clientes B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao detalhar clientes: {str(e)}")
        return clientes_lista

def consultar_taxas_software_b2b():
    prompt = """
No mercado brasileiro de software B2B:

1. Qual é a taxa média de conversão de prospecções em oportunidades comerciais?
2. Qual é a taxa média de conversão de oportunidades em vendas fechadas?

Forneça os dois valores em porcentagem. Exemplo:
- Prospecção para Oportunidade: 10%
- Oportunidade para Venda: 20%
"""
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em dados de mercado e vendas B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        texto = resposta.choices[0].message.content.strip()
        logger.info(f"Resposta OpenAI sobre conversão: {texto}")

        import re
        matches = re.findall(r"(\d{1,3})%", texto)

        if len(matches) >= 2:
            taxa_prospeccao = int(matches[0]) / 100
            taxa_vendas = int(matches[1]) / 100
            return taxa_prospeccao, taxa_vendas
        else:
            logger.warning("Não foi possível extrair as duas taxas. Usando padrão.")
            return 0.1, 0.2
    except Exception as e:
        logger.error(f"Erro ao consultar taxas de conversão: {str(e)}")
        return 0.1, 0.2

def buscar_conhecimento_complementado(pergunta: str) -> str:
    """Consulta o RAG e complementa com a OpenAI para respostas mais completas"""
    try:
        resposta_rag = buscar_conhecimento(pergunta, k=3)

        prompt = f"""Você é um consultor de canais de vendas B2B. Responda à pergunta abaixo de forma clara, objetiva e baseada em boas práticas e exemplos do mercado brasileiro.

Pergunta: {pergunta}

A resposta deve ser complementar a este conteúdo já obtido:
{resposta_rag}
"""
        resposta_openai = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor de canais de vendas com experiência em parcerias B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        return f"{resposta_rag}\n\n🔎 Complemento da IA:\n{resposta_openai.choices[0].message.content.strip()}"
    except Exception as e:
        logger.warning(f"Erro ao complementar conhecimento com OpenAI: {str(e)}")
        return buscar_conhecimento(pergunta)
