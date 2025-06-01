from fastapi import FastAPI, Request
from memory import perfil_prospect
from prompts import perguntas_estrategicas

app = FastAPI()

@app.get("/")
def home():
    return {"mensagem": "Agente de Canais AC Partners - Online"}

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    user_input = body.get("mensagem")

    for campo, pergunta in perguntas_estrategicas.items():
        if not perfil_prospect[campo]:
            perfil_prospect[campo] = user_input
            proxima = get_proxima_pergunta()
            return {"pergunta": proxima}

    return {
        "mensagem_final": "Todas as informações foram coletadas! Agora vamos iniciar os 12 passos da estratégia AC Partners."
    }

def get_proxima_pergunta():
    for campo, pergunta in perguntas_estrategicas.items():
        if not perfil_prospect[campo]:
            return pergunta
    return None


