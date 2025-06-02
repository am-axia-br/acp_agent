from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import os
import openai
import traceback
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
    "iniciado": False
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
        return {"mensagem": "Olá! Sou o Agente ACP, especialista em canais de vendas. Para começarmos, qual o seu nome?"}

    if not data["nome"]:
        data["nome"] = msg
        return {"pergunta": "Qual o nome da sua empresa?"}

    if not data["empresa"]:
        data["empresa"] = msg
        return {"pergunta": "Qual o seu WhatsApp? (Ex: 11 9 1234-5678)"}

    if not data["whatsapp"]:
        if re.match(r"^\d{2}\s9\s\d{4}-\d{4}$", msg):
            data["whatsapp"] = msg
            return {"pergunta": "Qual o seu e-mail?"}
        return {"pergunta": "Formato inválido. Exemplo: 11 9 1234-5678"}

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
                resposta = chamar_llm(prompt)
                data["finalizado"] = True
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

    return {"mensagem": "Diagnóstico já concluído."}

def gerar_prompt(data):
    blocos = "\n".join([f"{i+1}) {perguntas[i]} {resp}" for i, resp in enumerate(data["diagnostico"])]).strip()
    return f"""
Você é um especialista em canais de vendas no Brasil. Com base nas informações abaixo, analise o perfil da empresa respondente e forneça:

- Sugestões de modelos ideais de canais de vendas para esse tipo de empresa
- Perfis ideais de parceiros
- As 20 melhores cidades e regiões para buscar parceiros comerciais
- Projeção de faturamento com 20 canais ativos
- Quais tipos de apoio e recursos essa empresa pode oferecer aos parceiros

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
            {"role": "system", "content": "Você é um consultor especialista em canais de vendas."},
            {"role": "user", "content": prompt}
        ]
    )
    texto = resposta.choices[0].message.content.strip()
    linhas = texto.replace('. ', '.\n')
    enviar_email(data, linhas)
    return linhas



