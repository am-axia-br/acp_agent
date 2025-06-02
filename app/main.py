from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import os
import openai
import traceback
import json
from dotenv import load_dotenv
from mail import enviar_email
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/chat")
def chat_get():
    return {"erro": "Método GET não permitido nesta rota. Use POST com corpo JSON contendo 'mensagem'."}

class Mensagem(BaseModel):
    mensagem: str

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

perguntas = [
    "Qual o site da empresa?",
    "Quais segmentos a empresa atende?",
    "Poderia citar 3 clientes atuais?",
    "Quais dores a sua empresa resolve e quais benefícios são levados aos clientes?",
    "Quais os seus diferenciais?",
    "Quais os produtos e serviços vendidos?",
    "Qual o modelo de negócio da empresa? Comercializa licenças? Cobra mensalidade? Cobra projeto?",
    "Qual o ticket médio dos negócios?"
]

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    msg = body.get("mensagem", "").strip()

    if not data["iniciado"]:
        data["iniciado"] = True
        return {"mensagem": "Olá... Eu sou uma IA especialista em canais de vendas e o meu objetivo é te apresentar uma estratégia para você escalar suas vendas através de alianças, parcerias e canais de vendas... Vou te fazer algumas perguntas de forma que eu tenha as informações necessárias para gerar um diagnóstico completo e todas as ações que você deverá tomar. Para começarmos, me fale o seu nome..."}

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
        return {"pergunta": "Formato inválido. Exemplo: DDD9XXXXYYYY"}

    if not data["email"]:
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,4}$", msg):
            data["email"] = msg
            return {"mensagem": "Obrigado! Agora vamos entender melhor sua empresa.", "pergunta": perguntas[0]}
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
                return {
                    "mensagem": "Analisando as respostas e preparando o seu diagnóstico...",
                    "loading": True
                }
            except Exception as e:
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
        return {
            "mensagem": "Diagnóstico finalizado! Aqui está nossa análise baseada nas suas respostas:",
            "resumo": resposta,
            "email": data["email"]
        }
    except Exception as e:
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
    return {"status": "resetado"}

def gerar_prompt(data):
    blocos = "\n".join([f"{i+1}) {perguntas[i]} {resp}" for i, resp in enumerate(data["diagnostico"])]).strip()
    return f"""
Você é um especialista em canais de vendas no Brasil. Com base nas informações abaixo, analise o perfil da empresa respondente e forneça:

1. Sugestões de modelos ideais de canais de vendas para esse tipo de empresa
2. Perfis ideais de empresas para parcerias, alianças e canais de vendas
3. As 20 cidades brasileiras com maior potencial para atuação da empresa {data['empresa']}, apresentando:
   - Nome da cidade
   - População Estimada
   - PIB Atual
   - Número de empresas nos segmentos de atuação da {data['empresa']}
   - Segmento Econômico Principal
   - Número estimado de empresas com Perfil para Parceria com a {data['empresa']}, com base nos modelos indicados
4. Projeção de faturamento com 20 canais ativos
5. Quais tipos de apoio e recursos essa empresa pode oferecer aos parceiros comerciais

Dados do diagnóstico:
Nome: {data['nome']}
Empresa: {data['empresa']}
WhatsApp: {data['whatsapp']}
E-mail: {data['email']}

Respostas:
{blocos}
"""

def chamar_llm(prompt):
    resposta = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um consultor especialista em canais de vendas com acesso a uma base real de dados de cidades brasileiras. Use informações verificadas para compor sua resposta."},
            {"role": "user", "content": prompt}
        ]
    )
    texto = resposta.choices[0].message.content.strip()
    linhas = texto.replace('. ', '.\n')
    enviar_email(data, linhas)
    return linhas

