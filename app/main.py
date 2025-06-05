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
from rag_engine import filtrar_municipios_por_segmento, gerar_tabela_html
from rag_parcerias import buscar_conhecimento

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Aplicação FastAPI iniciada")

# 🔁 Reindexar automaticamente ao iniciar
@app.on_event("startup")
def indexar_automaticamente():
    try:
        from rag_parcerias import indexar_documentos
        total = indexar_documentos()
        logger.info(f"Indexação automática no startup concluída com {total} arquivos.")
    except Exception as e:
        logger.warning(f"Indexação automática ignorada: {e}")

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
    logger.info("Página inicial acessada e dados resetados")
    return FileResponse("static/index.html")

@app.get("/chat")
def chat_get():
    return {"erro": "Método GET não permitido nesta rota. Use POST com corpo JSON contendo 'mensagem'."}

class Mensagem(BaseModel):
    mensagem: str

perguntas = [
    "Qual o site da empresa?",
    "Quais segmentos a empresa atende?",
    "Poderia citar 3 clientes atuais?",
    "Quais dores a sua empresa resolve e quais benefícios são levados aos clientes?",
    "Quais os seus diferenciais?",
    "Quais os produtos e serviços vendidos?",
    "Qual o modelo de negócio da empresa? Comercializa licenças? Cobra mensalidade? Cobra projeto?",
    "Qual o ticket médio dos negócios?",
    "Qual o ciclo médio de vendas (em dias)?",
    "Qual a sua expectativa de vendas de novos clientes pelos canais mensalmente?"
]

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    msg = body.get("mensagem", "").strip()
    logger.info(f"Mensagem recebida: {msg}")

    if not data["iniciado"]:
        data["iniciado"] = True
        logger.info("Início do diagnóstico iniciado")
        return {"mensagem": "Olá... Para começarmos o diagnóstico, me fale o seu nome..."}

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
        logger.warning("WhatsApp em formato inválido")
        return {"pergunta": "Formato inválido. Exemplo: DDD9XXXXYYYY"}

    if not data["email"]:
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,4}$", msg):
            data["email"] = msg
            return {"mensagem": "Obrigado! Agora vamos entender melhor sua empresa.", "pergunta": perguntas[0]}
        logger.warning("E-mail inválido")
        return {"pergunta": "E-mail inválido. Envie um formato válido."}

    if data["etapa_atual"] < len(perguntas):
        data["diagnostico"].append(msg)
        data["etapa_atual"] += 1
        if data["etapa_atual"] < len(perguntas):
            return {"pergunta": perguntas[data["etapa_atual"]]}
        else:
            try:
                prompt = gerar_prompt(data)
                data["prompt"] = prompt
                data["finalizado"] = True
                logger.info("Diagnóstico preparado com sucesso")
                return {"mensagem": "Analisando as respostas e preparando o seu diagnóstico...", "loading": True}
            except Exception as e:
                logger.error(f"Erro ao preparar diagnóstico: {str(e)}")
                return {
                    "mensagem": "Ocorreu um erro ao preparar o diagnóstico.",
                    "resumo": f"Erro: {str(e)}\n\n{traceback.format_exc()}",
                    "email": data["email"]
                }

    return {"mensagem": "Diagnóstico já concluído."}

@app.post("/gerar-diagnostico")
async def gerar_diagnostico():
    try:
        resposta = chamar_llm(data["prompt"])
        logger.info("Diagnóstico gerado com sucesso pela LLM")
        return {
            "mensagem": "Diagnóstico finalizado! Aqui está nossa análise baseada nas suas respostas:",
            "resumo": resposta,
            "email": data["email"]
        }
    except Exception as e:
        logger.error(f"Erro ao gerar diagnóstico: {str(e)}")
        return {
            "mensagem": "Ocorreu um erro ao gerar o diagnóstico.",
            "resumo": f"Erro ao gerar sugestão: {str(e)}\n\n{traceback.format_exc()}",
            "email": data["email"]
        }

@app.post("/reset")
async def resetar_diagnostico():
    global data
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
    logger.info("Diagnóstico resetado pelo usuário")
    return {"status": "resetado"}

@app.post("/reindexar-rag")
async def reindexar_rag():
    try:
        from rag_parcerias import indexar_documentos
        total = indexar_documentos()
        logger.info(f"Reindexação realizada com {total} arquivos")
        return {"status": "ok", "arquivos_indexados": total}
    except Exception as e:
        logger.error(f"Erro na reindexação do RAG: {str(e)}")
        return {
            "status": "erro",
            "mensagem": str(e),
            "detalhes": traceback.format_exc()
        }

def gerar_prompt(data):
    bloco_respostas = "<ul>"
    for i, resp in enumerate(data["diagnostico"]):
        bloco_respostas += f"<li><strong>{i+1}) {perguntas[i]}</strong><br>{resp}</li>"
    bloco_respostas += "</ul>"

    segmento = data["diagnostico"][1] if len(data["diagnostico"]) > 1 else ""
    cidades_df = filtrar_municipios_por_segmento(segmento, top_n=30)

    if cidades_df.shape[0] < 30:
        faltando = 30 - cidades_df.shape[0]
        cidades_fake = [f"CidadeGenérica{i+1}" for i in range(faltando)]
        for cid in cidades_fake:
            cidades_df.loc[len(cidades_df)] = [cid, 100000, 10.0, 500, 100, 3000.0]

    cidades_html = gerar_tabela_html(cidades_df)

    try:
        raw_ticket = str(data["diagnostico"][7]).strip().replace("R$", "").replace(",", "").replace(".", "")
        raw_ciclo = str(data["diagnostico"][8]).strip()
        raw_novos = str(data["diagnostico"][9]).strip()

        if not raw_ticket.isdigit() or not raw_ciclo.isdigit() or not raw_novos.isdigit():
            raise ValueError(f"Valores inválidos recebidos: ticket={raw_ticket}, ciclo={raw_ciclo}, novos={raw_novos}")

        ticket = float(raw_ticket)
        ciclo = int(raw_ciclo)
        novos_clientes = int(raw_novos)
    except Exception as e:
        logger.error("Erro ao processar valores numéricos no prompt")
        raise ValueError(
            f"Erro ao converter valores numéricos: {str(e)} | "
            f"ticket={data['diagnostico'][7]}, ciclo={data['diagnostico'][8]}, novos={data['diagnostico'][9]}"
        ) from e

    conhecimento_parcerias = buscar_conhecimento("modelos de canais de vendas para empresas B2B")

    return f"""
<h2>Resumo do Diagnóstico Comercial</h2>
<p>Olá, {data['nome']},</p>

<p>Com base nas suas respostas, desenvolvemos abaixo o diagnóstico detalhado para sua empresa:</p>

<h3>Análise de Canais de Vendas para {data['empresa']}</h3>
<p>Veja abaixo os dados analisados:</p>
{bloco_respostas}

<h3>Modelos Ideais de Canais de Vendas</h3>
{conhecimento_parcerias}

<h3>📊 Cidades com maior potencial para parcerias</h3>
{cidades_html}

<h3>📈 Projeção de Resultados com 20 Canais Ativos</h3>
<ul>
  <li><strong>Ticket Médio:</strong> R${ticket:,.2f}</li>
  <li><strong>Meta Mensal de Novos Clientes por Canal:</strong> {novos_clientes} cliente(s)</li>
  <li><strong>Total de Novos Clientes com 20 Canais:</strong> {novos_clientes * 20} clientes</li>
  <li><strong>Receita Mensal Estimada:</strong> R${ticket * novos_clientes * 20:,.2f}</li>
</ul>

<h3>🔍 Cálculo de Oportunidades e Prospecções por Canal</h3>
<p><strong>Premissas:</strong> Conversão média do setor: 20%</p>
<ul>
  <li><strong>Oportunidades Necessárias por Canal:</strong> {int(novos_clientes / 0.2)} oportunidades</li>
  <li><strong>Prospecções Necessárias por Canal:</strong> {int((novos_clientes / 0.2) / 0.2)} prospecções</li>
</ul>

<p>Este diagnóstico fornece um panorama inicial para a expansão da sua empresa por meio de canais de vendas. Nossa recomendação é iniciar o onboarding com 3 a 5 canais para validação.</p>
"""

def chamar_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor especialista em canais de vendas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Erro na chamada à API OpenAI")
        raise RuntimeError("Erro na chamada à API OpenAI.") from e
