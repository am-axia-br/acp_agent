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
    blocos = "\n".join([f"{i+1}) {perguntas[i]} {resp}" for i, resp in enumerate(data["diagnostico"])])
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
            f"ticket={data['diagnostico'][7]}, ciclo={data['diagnostico'][8]}, novos={data['diagnostico'][9]}") from e

    conhecimento_parcerias = buscar_conhecimento("modelos de canais de vendas para empresas B2B")
    conhecimento_formatado = f"\n### Base de Conhecimento sobre Parcerias:\n\n{conhecimento_parcerias}\n\n"

    return f"""
Você é um consultor especialista em canais de vendas. Use os dados do cliente e o conhecimento abaixo para gerar um diagnóstico estruturado com os seguintes tópicos:

{conhecimento_formatado}

01) Resumo sobre a empresa. Pesquise no site informado e use dados disponíveis na internet.
02) Situação do mercado e perfil dos clientes que essa empresa atende.
03) Oportunidades de crescimento e expansão da empresa com canais de vendas.
04) Liste 5 modelos ideais de canais de vendas com explicações de funcionamento, vantagens e serviços agregados.
05) Descreva os perfis ideais de empresas que podem se tornar canais de vendas.
06) Liste 30 cidades com maior potencial para abertura de canais (dados: nome, população, PIB, empresas no segmento, empresas com perfil de canal, salário médio).

<h3 style='color:#5e17eb;margin-top:30px;'>📍 Cidades com Potencial</h3>
{cidades_html}

07) Faça um cálculo de retorno financeiro com 20 canais ativos, assumindo o ticket médio informado.
Além disso, calcule:
- Quantas oportunidades por canal são necessárias para atingir a meta mensal de novos clientes.
- Quantas prospecções são necessárias por canal com base no índice médio de conversão do setor da empresa (pesquise esse índice).

Dados:
- Ticket médio: R${ticket:,.2f}
- Ciclo médio de vendas: {ciclo} dias
- Meta mensal de novos clientes por canal: {novos_clientes}

Dados do diagnóstico:
Nome: {data['nome']}
Empresa: {data['empresa']}
WhatsApp: {data['whatsapp']}
E-mail: {data['email']}

Respostas:
{blocos}
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

