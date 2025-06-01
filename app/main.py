from fastapi import FastAPI, Request
from pydantic import BaseModel, EmailStr
import re

app = FastAPI()

# Armazena o estado da conversa e os dados
respostas = {
    "nome": None,
    "empresa": None,
    "whatsapp": None,
    "email": None,
    "diagnostico": [],
    "etapa_atual": 0
}

# Lista de perguntas do diagnóstico
perguntas_diagnostico = [
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

class Mensagem(BaseModel):
    mensagem: str

@app.get("/")
def home():
    return {"mensagem": "Agente de Canais AC Partners - Online"}

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    mensagem = body.get("mensagem", "").strip()

    # Coleta nome
    if not respostas["nome"]:
        respostas["nome"] = mensagem
        return {"pergunta": "Qual o nome da sua empresa?"}

    # Coleta empresa
    if not respostas["empresa"]:
        respostas["empresa"] = mensagem
        return {"pergunta": "Qual o seu WhatsApp? (Ex: 11 9 1234-5678)"}

    # Valida WhatsApp
    if not respostas["whatsapp"]:
        if re.match(r"^\d{2}\s9\s\d{4}-\d{4}$", mensagem):
            respostas["whatsapp"] = mensagem
            return {"pergunta": "Qual o seu e-mail?"}
        else:
            return {"pergunta": "Formato de WhatsApp inválido. Exemplo: 11 9 1234-5678"}

    # Valida e-mail
    if not respostas["email"]:
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,4}$", mensagem):
            respostas["email"] = mensagem
            return {"mensagem": "Obrigado pelas informações! Agora vamos entender melhor a situação atual da sua empresa.", "pergunta": perguntas_diagnostico[0]}
        else:
            return {"pergunta": "E-mail inválido. Por favor, envie um e-mail válido."}

    # Diagnóstico: perguntas 1 a 16
    if respostas["etapa_atual"] < len(perguntas_diagnostico):
        respostas["diagnostico"].append(mensagem)
        respostas["etapa_atual"] += 1
        if respostas["etapa_atual"] < len(perguntas_diagnostico):
            return {"pergunta": perguntas_diagnostico[respostas["etapa_atual"]]}
        else:
            return {
                "mensagem": "Diagnóstico finalizado! Com base nas suas respostas, irei gerar uma sugestão de estratégia de canais:",
                "resumo": {
                    "nome": respostas["nome"],
                    "empresa": respostas["empresa"],
                    "whatsapp": respostas["whatsapp"],
                    "email": respostas["email"],
                    "respostas": respostas["diagnostico"]
                },
                "proximos_passos": [
                    "Modelos de canais ideais",
                    "Perfis de parceiros recomendados",
                    "Top 20 cidades para expansão",
                    "Projeção de faturamento com 20 canais"
                ]
            }
