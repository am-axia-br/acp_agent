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
from rag_engine import filtrar_municipios_por_segmentos_multiplos as filtrar_municipios_por_segmento, gerar_tabela_html, normalizar_segmentos
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


def novo_diagnostico():
    return {
        "origem": None,
        "nome": None,
        "empresa": None,
        "whatsapp": None,
        "email": None,
        "diagnostico": [],
        "etapa_atual": 0,
        "finalizado": False,
        "iniciado": False,
        "prompt": None
    }

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

data = novo_diagnostico


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
                logger.info("Diagnostico preparado com sucesso")
                return {"mensagem": "Analisando as respostas e preparando o seu diagnostico...", "loading": True}
            except Exception as e:
                logger.error(f"Erro ao preparar diagnostico: {str(e)}")
                return {"mensagem": "Ocorreu um erro ao preparar o diagnostico.", "resumo": f"Erro: {str(e)}\n\n{traceback.format_exc()}", "email": data["email"]}

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

    # Variáveis nomeadas para facilitar a leitura e permitir uso em perguntas dinâmicas
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

    segmento_original = segmentos_raw if len(respostas) > 1 else ""
    segmentos_normalizados = normalizar_segmentos(segmento_original)

    segmento = truncar_texto(segmentos_raw)
    clientes = truncar_texto(clientes)
    dores = truncar_texto(dores)
    produtos = truncar_texto(produtos)
    modelo = truncar_texto(modelo_negocio)

    conhecimento_modelos = buscar_conhecimento(f"modelos de canais para o segmento {segmento}")
    conhecimento_perfis = buscar_conhecimento(f"empresas ideais para parcerias em {segmento} como as atendidas ({clientes})")
    conhecimento_servicos = buscar_conhecimento(f"servicos agregados relevantes para empresas que vendem {produtos} com modelo de negocio {modelo}")

    segmentos_str = ", ".join(segmentos_normalizados)
    cidades_df = filtrar_municipios_por_segmento(segmentos_str, top_n=30)

    for col in ["Municipio", "Populacao", "PIB", "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"]:
        if col not in cidades_df.columns:
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    if len(cidades_df) < 30:
        cidades_existentes = cidades_df["Municipio"].tolist() if "Municipio" in cidades_df.columns else []
        faltantes = 30 - len(cidades_existentes)
        sugestoes = sugerir_cidades_openai(segmentos_normalizados, total_necessario=faltantes)

        cidades_extra = [c for c in sugestoes if c not in cidades_existentes]
        cidades_completas = cidades_existentes + cidades_extra[:faltantes]

        if faltantes > 0 and cidades_extra:
            cidades_df_extra = pd.DataFrame({
                "Municipio": cidades_extra[:faltantes],
                "Populacao": [0] * faltantes,
                "PIB": [0] * faltantes,
                "Empresas_Segmento": [0] * faltantes,
                "Empresas_Perfil_Canal": [0] * faltantes,
                "Salario_Medio_R$": [0] * faltantes
            })
            cidades_df = pd.concat([cidades_df, cidades_df_extra], ignore_index=True)

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

    # Novo cálculo: Receita acumulada com canais ativando 1 por mês
    from datetime import datetime, timedelta
    receita_24_meses = 0
    hoje = datetime.today()

    for canal_index in range(20):  # Máximo de 20 canais
        dias_ate_primeira_venda = 90 + ciclo + (canal_index * 30)
        data_primeira_venda = hoje + timedelta(days=dias_ate_primeira_venda)
        meses_restantes = max(0, 24 - ((data_primeira_venda - hoje).days // 30))
        receita_24_meses += ticket * meses_restantes

    receita_mensal = ticket * novos_clientes * 20  # Receita "cheia" hipotética

    return f'''
🧠 DIAGNÓSTICO ESTRUTURADO – EMPRESA {empresa.upper()}

🔹 Local informado:
{origem}

🔹 Parte 01 – Resumo sobre a empresa:
{site}

🔹 Parte 02 – Produtos e Serviços:
{produtos}

🔹 Parte 03 – Perfil dos Clientes atendidos:
{clientes}

🔹 Parte 04 – Dores e Benefícios:
{dores}

🔹 Parte 05 – Modelo de Negócio:
{modelo_negocio}

🔹 Parte 06 – Necessidades de Prospecção:
Meta Prevista: {novos_clientes} novos clientes/mês
Ciclo de Venda: {ciclo} dias
Índice de Conversão: {int(conversao * 100)}%
Número de Oportunidades: {oportunidades}
Número de Prospecções: {prospeccoes}

🔹 Parte 07 – Modelos de Parceria:
{conhecimento_modelos}

🔹 Parte 08 – Perfis de Empresas Parceiras com FIT:
{conhecimento_perfis}

🔹 Parte 09 – Serviços Agregados:
{conhecimento_servicos}

🔹 Parte 10 – Cidades com Potencial:
{cidades_html}

🔹 Parte 11 – Retorno sobre o Investimento:
Ticket Médio: R${ticket:,.2f}
Receita Mensal com 20 canais: R${receita_mensal:,.2f}
Receita estimada em 24 meses: R${receita_24_meses:,.2f}

🔹 Parte 12 – Dicas Estratégicas:
- Inicie com 3 a 5 canais
- Forneça treinamentos e acompanhamento semanal
- Crie campanhas e ações de marketing
- Corrija falhas com feedback dos canais

🔹 Parte 13 – Chamada para Ação:
Sua empresa está pronta para crescer com uma estratégia sólida de canais de vendas.
Entre em contato com a AC Partners e comece agora o onboarding comercial com especialistas.
'''


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
            max_tokens=800
        )
        cidades = resposta.choices[0].message.content.strip().split("\n")
        return [c.strip().split("-")[0].strip() for c in cidades if c.strip()]
    except Exception as e:
        logger.error(f"Erro ao sugerir cidades com OpenAI: {str(e)}")
        return []

