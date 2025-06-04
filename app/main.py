from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import os
import openai
import traceback
import json
import unicodedata
from dotenv import load_dotenv
from mail import enviar_email
from openai import OpenAI
from rag_engine import filtrar_municipios_por_segmento, gerar_tabela_html

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

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
    "Qual a meta mensal de vendas esperada por canal (em R$)?"
]

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    msg = body.get("mensagem", "").strip()

    if not data["iniciado"]:
        data["iniciado"] = True
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
                return {"mensagem": "Analisando as respostas e preparando o seu diagnóstico...", "loading": True}
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
        ticket = float(data["diagnostico"][7].replace("R$", "").replace(",", "").strip())
        ciclo = int(data["diagnostico"][8])
        meta_mensal = float(data["diagnostico"][9].replace("R$", "").replace(",", "").strip())
    except:
        ticket = ciclo = meta_mensal = 0

    return f"""
Você é um consultor especialista em canais de vendas. Gere um diagnóstico estruturado com os seguintes tópicos:

01) Resumo sobre a empresa. Pesquise no site informado e use dados disponíveis na internet.

02) Situação do mercado e perfil dos clientes que essa empresa atende.

03) Oportunidades de crescimento e expansão da empresa com canais de vendas.

04) Liste 5 modelos ideais de canais de vendas com explicações de funcionamento, vantagens e serviços agregados.

05) Descreva os perfis ideais de empresas que podem se tornar canais de vendas.

06) Liste 30 cidades com maior potencial para abertura de canais (dados: nome, população, PIB, empresas no segmento, empresas com perfil de canal, salário médio). Se o RAG não retornar 30, complemente com sugestões próprias.

{cidades_html}

07) Faça um cálculo de retorno financeiro com 20 canais ativos, assumindo o ticket médio informado.

Além disso, calcule:
- Quantas oportunidades por canal são necessárias para atingir a meta mensal.
- Quantas prospecções são necessárias por canal com base no índice médio de conversão do setor da empresa (pesquise esse índice).

Dados:
- Ticket médio: R${ticket:,.2f}
- Ciclo médio de vendas: {ciclo} dias
- Meta mensal por canal: R${meta_mensal:,.2f}

Dados do diagnóstico:
Nome: {data['nome']}
Empresa: {data['empresa']}
WhatsApp: {data['whatsapp']}
E-mail: {data['email']}

Respostas:
{blocos}
"""

def limpar_unicode(texto_raw: str) -> str:
    texto_limpo = texto_raw.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    texto_utf8 = texto_limpo.encode("utf-8", "replace").decode("utf-8", "replace")
    return unicodedata.normalize("NFC", texto_utf8)

def remover_surrogates(texto: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', texto)

def chamar_llm(prompt):
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um consultor especialista em canais de vendas com acesso a uma base real de dados sobre cidades brasileiras. Use sempre dados plausíveis e consistentes, formatados em HTML limpo, espaçado e sem asteriscos ou hashtags. Use <h2>, <h3>, <p> e espaçamento visual elegante. Separe cada seção com exatamente duas linhas em branco."
                },
                {"role": "user", "content": prompt}
            ]
        )
        texto_original = resposta.choices[0].message.content
        texto = remover_surrogates(limpar_unicode(texto_original.strip())).replace("**", "")

        linhas_formatadas = []
        for linha in texto.split("\n"):
            linha = linha.strip()
            if not linha:
                continue
            elif linha.lower().startswith("### "):
                linhas_formatadas.append(f"<h2 style='color:#5e17eb;margin-top:30px;'>{linha[4:]}</h2>\n\n")
            elif linha.lower().startswith("## "):
                linhas_formatadas.append(f"<h3 style='color:#a638ec;margin-top:20px;'>{linha[3:]}</h3>\n\n")
            elif linha.lower().startswith("# "):
                linhas_formatadas.append(f"<h4 style='color:#fc6736;margin-top:15px;'>{linha[2:]}</h4>\n\n")
            else:
                linhas_formatadas.append(f"<p style='margin-bottom:15px'>{linha}</p>\n\n")

        html_formatado = "\n".join(linhas_formatadas)
        enviar_email(data, html_formatado)
        return html_formatado

    except UnicodeEncodeError as e:
        return f"Erro de codificação ao gerar sugestão: {str(e)}"
    except Exception as e:
        return f"Erro inesperado ao chamar LLM: {str(e)}"
