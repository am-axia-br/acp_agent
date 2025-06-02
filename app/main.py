from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import os
import openai
from dotenv import load_dotenv
from mail import enviar_email

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Servir arquivos estaticos e index.html
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
    "Quais os segmentos a empresa atende?",
    "Quais as características dos clientes da empresa?",
    "Poderia citar 3 clientes atuais?",
    "Quais as dores e os benefícios são levados aos clientes?",
    "Quais os seus diferenciais?",
    "Quais os produtos e serviços vendidos?",
    "Como é o processo comercial atual?",
    "Qual o modelo de negócio da empresa? Comercializa licenças? Cobra mensalidade? Cobra projeto?",
    "Qual o ticket médio das mensalidades?",
    "Qual o ticket médio dos projetos?",
    "Você possui indicadores comerciais? Quais?",
    "Como é o seu processo comercial atual?",
    "Você tem experiência com canais de vendas? Me fale desta experiência!",
    "Você tem uma proposta de valor já desenvolvida?",
    "Você tem uma meta mensal de vendas? Quais são estas metas?"
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
            prompt = gerar_prompt(data)
            resposta = chamar_llm(prompt)
            data["finalizado"] = True
            enviar_email(data, resposta)
            return {
                "mensagem": "Diagnóstico finalizado! Aqui está nossa análise baseada nas suas respostas:",
                "resumo": resposta,
                "email": data["email"]
            }

    return {"mensagem": "Diagnóstico já concluído."}

def gerar_prompt(data):
    blocos = "\n".join([f"{i+1}) {perguntas[i]} {resp}" for i, resp in enumerate(data["diagnostico"])])
    return f"""
Você é um especialista em canais de vendas no Brasil.
Com base nas respostas abaixo, forneça:
- Melhores modelos de canais
- Perfis ideais de parceiros
- As 20 melhores cidades e regiões para canais
- Projeção de faturamento com 20 canais ativos

Informações:
Nome: {data['nome']}
Empresa: {data['empresa']}
WhatsApp: {data['whatsapp']}
E-mail: {data['email']}

Respostas:
{blocos}
"""

def chamar_llm(prompt):
    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um consultor especialista em canais de vendas."},
                {"role": "user", "content": prompt}
            ]
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar sugestão: {str(e)}"
