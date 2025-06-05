from log_config import get_logger
logger = get_logger(__name__)

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import os
import traceback
import json
from dotenv import load_dotenv
from mail import enviar_email
from openai import OpenAI
from rag_engine import filtrar_municipios_por_segmentos_multiplos as filtrar_municipios_por_segmento, gerar_tabela_html, normalizar_segmentos
from rag_parcerias import buscar_conhecimento

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

data = {
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

@app.get("/")
def home():
    global data
    data = {"nome": None, "empresa": None, "whatsapp": None, "email": None, "diagnostico": [], "etapa_atual": 0, "finalizado": False, "iniciado": False, "prompt": None}
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
        logger.info("Inicio do diagnostico iniciado")
        return {"mensagem": "Ola... Para comecarmos o diagnostico, me fale o seu nome..."}

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
        data["etapa_atual"] += 1
        if data["etapa_atual"] < len(perguntas):
            return {"pergunta": perguntas[data["etapa_atual"]]}
        else:
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
    data = {"nome": None, "empresa": None, "whatsapp": None, "email": None, "diagnostico": [], "etapa_atual": 0, "finalizado": False, "iniciado": False, "prompt": None}
    logger.info("Diagnostico resetado pelo usuario")
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

def gerar_prompt(data):
    nome = data["nome"]
    empresa = data["empresa"]
    segmento_original = data["diagnostico"][1] if len(data["diagnostico"]) > 1 else ""
    segmentos_normalizados = normalizar_segmentos(segmento_original)

    bloco_respostas = ""
    for i, resp in enumerate(data["diagnostico"]):
        bloco_respostas += f"{i+1}) {perguntas[i]} \n{resp}\n\n"

    conhecimento_modelos = buscar_conhecimento("modelos de canais de vendas para empresas B2B")
    conhecimento_perfis = buscar_conhecimento("perfis e tipos ideais de canais de vendas")
    conhecimento_servicos = buscar_conhecimento("servicos agregados em canais de vendas")

    cidades_df = filtrar_municipios_por_segmento(segmentos_normalizados, top_n=30)
    if cidades_df.shape[0] < 30:
        faltando = 30 - cidades_df.shape[0]
        cidades_fake = [f"CidadeGenerica{i+1}" for i in range(faltando)]
        for cid in cidades_fake:
            cidades_df.loc[len(cidades_df)] = [cid, 100000, 10.0, 500, 100, 3000.0]
    cidades_html = gerar_tabela_html(cidades_df)

    try:
        ticket = float(data["diagnostico"][7])
        ciclo = int(data["diagnostico"][8])
        novos_clientes = int(data["diagnostico"][9])
    except Exception as e:
        logger.error("Erro ao processar valores numericos no prompt")
        raise ValueError(f"Erro ao converter valores numericos: {str(e)} | ticket={data['diagnostico'][7]}, ciclo={data['diagnostico'][8]}, novos={data['diagnostico'][9]}") from e

    return f"""
Resumo do Diagnostico Comercial

Ola, {nome},

Com base nas suas respostas, desenvolvemos abaixo o diagnostico detalhado para sua empresa: {empresa}.

Dados analisados:
{bloco_respostas}

Modelos ideais de canais de vendas:
{conhecimento_modelos}

Tipos, perfis e perfis ideais:
{conhecimento_perfis}

Servicos agregados para canais:
{conhecimento_servicos}

{cidades_html}

Projecao de resultados com 20 canais ativos:
- Ticket Medio: R${ticket:,.2f}
- Meta Mensal de Novos Clientes por Canal: {novos_clientes} cliente(s)
- Total de Novos Clientes com 20 Canais: {novos_clientes * 20} clientes
- Receita Mensal Estimada: R${ticket * novos_clientes * 20:,.2f}

Calculo de oportunidades e prospeccoes por canal (base: 20% de conversao):
- Oportunidades Necessarias por Canal: {int(novos_clientes / 0.2)}
- Prospeccoes Necessarias por Canal: {int((novos_clientes / 0.2) / 0.2)}

Nossa recomendacao e iniciar o onboarding com 3 a 5 canais para validacao.
"""

def chamar_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Voce e um consultor especialista em canais de vendas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Erro na chamada a API OpenAI")
        raise RuntimeError("Erro na chamada a API OpenAI.") from e


