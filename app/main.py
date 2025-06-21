import os
import re
import logging
import traceback
import pandas as pd
from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from mail import enviar_email
from openai import OpenAI
# (demais imports do seu projeto)
from segmento_equivalencias import *

from segmento_equivalencias import buscar_segmentos_em_df_avancado, descricoes_cnae, embeddings_cnae


app = FastAPI()
df_cnae = pd.read_excel("Tabela 14.xlsx")


@app.get("/cnae/segmentos/")
def get_cnae_por_segmentos(segmentos: str = Query(..., description="Segmentos separados por vírgula")):
    termos = [s.strip() for s in segmentos.split(",")]
    df_filtrado = buscar_segmentos_em_df_avancado(
        df_cnae,
        ["Descrição"],
        termos,
        embeddings_cnae=embeddings_cnae,
        descricoes_cnae=descricoes_cnae,
        fuzzy_threshold=85,
        emb_threshold=0.7,
        use_embeddings=True
    )
    return df_filtrado.to_dict(orient="records")

# ====== NLTK seguro para Render ======
import nltk
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
# ================================================================

# ====== INÍCIO: NOVO BLOCO - LangChain RAG para planilhas =======

from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

def indexar_planilhas_rag():
    arquivos = ['Tabela 14.xlsx', 'canais.xlsx']
    documentos = []
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            df = pd.read_excel(arquivo)
            for idx, row in df.iterrows():
                texto = " | ".join([str(x) for x in row.values])
                documentos.append(
                    Document(page_content=texto, metadata={"source": arquivo, "linha": idx + 1})
                )
            print(f"{arquivo} lido com {df.shape[0]} linhas")
        else:
            print(f"Arquivo não encontrado: {arquivo}")

    if not documentos:
        print("Nenhum documento para indexar!")
        return None

    splitter = NLTKTextSplitter(chunk_size=5, chunk_overlap=1)
    chunks = splitter.split_documents(documentos)
    print(f"Total de chunks: {len(chunks)}")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    print("Indexação RAG concluída!")
    return db

db_rag = None  # VARIÁVEL GLOBAL para reuso se quiser usar buscas posteriormente

# ====== FIM: NOVO BLOCO LangChain RAG ===========================

import logging

STOPWORDS = {
    "para", "com", "sem", "de", "e", "ou", "por", "em", "da", "do",
    "no", "na", "das", "dos"
}

logger = logging.getLogger(__name__)

from rag_engine import (
    filtrar_municipios_por_segmentos_multiplos as filtrar_municipios_por_segmento,
    gerar_tabela_html,
    normalizar_segmentos_inteligente,
    descricoes_cnae,
    embeddings_cnae,
    buscar_cidades_na_openai,
)

from rag_parcerias import buscar_conhecimento

def consultar_taxa_conversao_openai(segmentos):
    """
    Consulta a taxa de conversão média para o(s) segmento(s) informado(s) usando a OpenAI.
    """
    try:
        prompt = (
            f"Para empresas do segmento {', '.join(segmentos)}, qual é a taxa média de conversão de:\n"
            "- Prospecção para oportunidade?\n"
            "- Oportunidade para venda?\n"
            "Dê um número percentual para cada uma, no Brasil, em mercados B2B."
        )
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
            return 0.2 if isinstance(segmentos, list) else 0.2
    except Exception as e:
        logger.error(f"Erro ao consultar taxa de conversao: {str(e)}")
        return 0.2 if isinstance(segmentos, list) else 0.2

def novo_diagnostico():
    """Gera um novo dicionário de diagnóstico padrão."""
    return {
        "nome": None,
        "empresa": None,
        "whatsapp": None,
        "email": None,
        "diagnostico": [],
        "etapa_atual": 0,
        "iniciado": False,
        "municipios": [],
        "resumo": "",
        "mensagem_usuario": "",
        "resposta_ia": ""
    }

# VARIÁVEL GLOBAL DO ESTADO DO DIAGNÓSTICO
data = novo_diagnostico()

# Inicialização do ambiente e OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instância do FastAPI e montagem dos arquivos estáticos

app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Aplicacao FastAPI iniciada")

@app.on_event("startup")
def indexar_automaticamente():
    global db_rag
    try:
        db_rag = indexar_planilhas_rag()  # NOVO: indexa as planilhas no deploy
        from rag_parcerias import indexar_documentos
        total = indexar_documentos()
        logger.info(f"Indexacao automatica no startup concluida com {total} arquivos.")
    except Exception as e:
        logger.warning(f"Indexacao automatica ignorada: {e}")

@app.get("/")
def home():
    """Endpoint padrão para servir a página inicial e resetar o diagnóstico."""
    global data
    data = novo_diagnostico()
    logger.info("Pagina inicial acessada e dados resetados")
    return FileResponse("static/index.html")

@app.get("/chat")
def chat_get():
    """Bloqueia acesso GET na rota de chat."""
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
    """
    Endpoint principal do chat. Recebe perguntas e coleta as informações para o diagnóstico.
    """
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
                return {
                    "mensagem": "Erro: Respostas incompletas. Por favor, reinicie o diagnóstico.",
                    "resumo": "Erro: respostas insuficientes.",
                    "email": data["email"]
                }
            try:
                prompt = gerar_prompt(data)
                data["prompt"] = prompt
                data["finalizado"] = True
                logger.info("Prompt preparado com sucesso")
                return {
                    "mensagem": "Analisando suas respostas e preparando o diagnóstico...",
                    "loading": True,
                    "email": data["email"]
                }
            except Exception as e:
                print(f"Erro ao preparar prompt: {e}")
                return {
                    "resumo": f"Erro técnico: {str(e)}"
                }

    return {"mensagem": "Diagnostico ja concluido."}

@app.post("/gerar-diagnostico")
async def gerar_diagnostico():
    try:
        diagnostico, cidades_html = gerar_prompt(data)  # <- agora retorna dois campos!

        #enviar_email(data, f"{diagnostico}\n\n{cidades_html}", copia_para=["alexandre.maia@acp.tec.br"])

        logger.info("Diagnostico gerado com sucesso e e-mail enviado")
        return {
            "mensagem": "Diagnóstico finalizado! Aqui está nossa análise baseada nas suas respostas:",
            "diagnostico": diagnostico,
            "cidades_html": cidades_html,
            "email": data["email"]
        }
    except Exception as e:
        logger.error(f"Erro ao gerar diagnostico: {str(e)}")
        return {
            "mensagem": "Ocorreu um erro ao gerar o diagnóstico.",
            "resumo": f"Erro ao gerar sugestão: {str(e)}\n\n{traceback.format_exc()}",
            "email": data["email"]
        }

@app.post("/enviar-email-final")
async def enviar_email_final(req: Request):
    body = await req.json()
    email = body.get("email")
    diagnostico = body.get("diagnostico")
    cidades_html = body.get("cidades_html")

    corpo_email = f"""
    <h2>Resumo do Diagnóstico</h2>
    {diagnostico}
    <h2>Ranking de Cidades com Potencial</h2>
    {cidades_html}
    """

    try:
        # Ajuste conforme sua função real de envio de e-mail
        enviar_email({"email": email}, corpo_email, copia_para=["alexandre.maia@acp.tec.br"])
        return {"status": "ok", "mensagem": "Diagnóstico enviado com sucesso!"}
    except Exception as e:
        return {"status": "erro", "mensagem": f"Erro ao enviar e-mail: {str(e)}"}

@app.post("/reset")
async def resetar_diagnostico():
    """Reseta o estado do diagnóstico global."""
    global data
    data = novo_diagnostico()
    return {"status": "resetado"}

@app.post("/reindexar-rag")
async def reindexar_rag():
    global db_rag
    try:
        db_rag = indexar_planilhas_rag()  # NOVO: reindexa as planilhas manualmente
        from rag_parcerias import indexar_documentos
        total = indexar_documentos()
        logger.info(f"Reindexacao realizada com {total} arquivos")
        return {"status": "ok", "arquivos_indexados": total}
    except Exception as e:
        logger.error(f"Erro na reindexacao do RAG: {str(e)}")
        return {
            "status": "erro",
            "mensagem": str(e),
            "detalhes": traceback.format_exc()
        }
    
def truncar_texto(texto, limite=500):
    """Trunca um texto se for maior do que o limite definido."""
    return texto[:limite] + "..." if len(texto) > limite else texto

def gerar_prompt(data):
    """
    Monta o prompt estruturado para análise do diagnóstico e geração do relatório.
    Inclui o bloco de cidades em HTML no local correto do texto final.
    """
    origem = data.get("origem", "")
    nome = data["nome"]
    empresa = data["empresa"]
    respostas = data["diagnostico"]




    if len(respostas) < 2:
        raise ValueError("Respostas insuficientes para gerar o prompt.")

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

    if isinstance(segmentos_raw, list):
        segmentos_raw = " ".join(segmentos_raw)

    segmento_original = segmentos_raw if len(respostas) > 1 else ""

    # Lista de segmentos limpa
    segmentos_lista = [
        seg.strip().lower()
        for seg in re.split(r"[,\s]+", segmento_original)
        if seg.strip().lower() not in STOPWORDS and len(seg.strip()) > 2
    ]

    segmentos_normalizados = []
    for termo in segmentos_lista:
        normalizados = normalizar_segmentos_inteligente(termo, descricoes_cnae, embeddings_cnae)
        segmentos_normalizados.extend(normalizados)

    segmento = truncar_texto(segmentos_raw)
    clientes = truncar_texto(clientes)
    dores = truncar_texto(dores)
    produtos = truncar_texto(produtos)
    modelo = truncar_texto(modelo_negocio)

    resumo_empresa = detalhar_empresa_openai(site, empresa, segmento, clientes, produtos, modelo)    
    resumo_produto = detalhar_produto_openai(produtos, segmento)
    resumo_clientes = detalhar_clientes_openai(clientes, ticket_medio_str, segmento, produtos, modelo)
    taxa_pros, taxa_vendas = consultar_taxas_software_b2b()

    try:
        novos_clientes = int(meta_clientes)
        ciclo = int(ciclo_vendas)
    except Exception:
        novos_clientes = 1
        ciclo = 90

    oportunidades = int(novos_clientes / taxa_vendas)
    prospeccoes = int(oportunidades / taxa_pros)

    conhecimento_modelos = buscar_modelos_parceria_empresa(
        nome_empresa=empresa,
        site=site,
        clientes=clientes,
        ticket_medio=ticket_medio_str,
        ciclo_vendas=ciclo_vendas,
        produto=produtos,
        segmentos=segmento,
        modelo_negocio=modelo_negocio,
        k=5
    )
    
    perfis_e_modelos = conhecimento_perfis(
        nome_empresa=empresa,
        site=site,
        clientes=clientes,
        ticket_medio=ticket_medio_str,
        ciclo_vendas=ciclo_vendas,
        produto=produtos,
        segmentos=segmento,
        modelo_negocio=modelo_negocio,
        k=5
    )


    servicos_agregados = relacao_servicos_agregados(
        nome_empresa=empresa,
        site=site,
        clientes=clientes,
        ticket_medio=ticket_medio_str,
        ciclo_vendas=ciclo_vendas,
        produto=produtos,
        segmentos=segmento,
        modelo_negocio=modelo_negocio,
        dores=dores,
        beneficios=diferenciais,
        diferenciais=diferenciais,
        k=10
    )

    dicas_estrategicas = gerar_dicas_estrategicas(
        site=site,
        segmentos=segmento,
        clientes=clientes,
        dores=dores,
        beneficios=diferenciais,
        diferenciais=diferenciais,
        produtos=produtos,
        modelo_negocio=modelo_negocio,
        ticket_medio=ticket_medio_str,
        ciclo_vendas=ciclo_vendas,
        meta_clientes=meta_clientes,
        modelos_parceria=conhecimento_modelos,
        perfis_canais=perfis_e_modelos,
        servicos_agregados=servicos_agregados
    )

    # Busca as cidades (RAG + OpenAI para completar 30)
    
    if isinstance(segmento_original, list):
        segmento_original = " ".join(segmento_original)
    cidades_df = filtrar_municipios_por_segmento(segmento_original, top_n=30)


    print("==== DEBUG cidades_df ====")
    print(cidades_df.head(10))
    print("==== FIM DEBUG cidades_df ====")


    for col in ["Empresas_Segmento", "Empresas_Perfil_Canal"]:
        if col in cidades_df.columns:
            cidades_df[col] = pd.to_numeric(cidades_df[col], errors="coerce").fillna(0).astype(int)

    if cidades_df.empty or any(col not in cidades_df.columns for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]):
        logger.warning("DataFrame de cidades incompleto. Criando estrutura padrão.")
        cidades_df = pd.DataFrame({
            "Municipio": ["CidadeDesconhecida"],
            "Empresas_Segmento": [0],
            "Empresas_Perfil_Canal": [0]
        })

    for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
        if col not in cidades_df.columns:
            cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"

    try:
        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"[FALHA] Erro ao ordenar DataFrame por 'Empresas_Segmento': {e}")

    if len(cidades_df) < 30:
        if cidades_df.empty or "Municipio" not in cidades_df.columns:
            cidades_df = pd.DataFrame({
                "Municipio": ["CidadeDesconhecida"],
                "Empresas_Segmento": [0],
                "Empresas_Perfil_Canal": [0]
            })
        cidades_existentes = cidades_df["Municipio"].tolist()
        faltantes = 30 - len(cidades_df)
        cidades_df_extra = buscar_cidades_na_openai(segmentos_lista, cidades_existentes, faltantes)
        for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
            if col not in cidades_df_extra.columns:
                cidades_df_extra[col] = 0 if col != "Municipio" else "CidadeDesconhecida"
        cidades_df = pd.concat([cidades_df, cidades_df_extra], ignore_index=True)
        cidades_df = cidades_df.drop_duplicates(subset="Municipio").reset_index(drop=True)
        faltam = 30 - len(cidades_df)
        if faltam > 0:
            cidades_df_extra = buscar_cidades_na_openai(
                segmentos_lista, cidades_df["Municipio"].tolist(), faltam
            )
            cidades_df = pd.concat(
                [cidades_df, cidades_df_extra], ignore_index=True
            ).drop_duplicates(subset="Municipio").reset_index(drop=True)
        for col in ["Municipio", "Empresas_Segmento", "Empresas_Perfil_Canal"]:
            if col not in cidades_df.columns:
                cidades_df[col] = 0 if col != "Municipio" else "CidadeDesconhecida"
        cidades_df = cidades_df.sort_values(by="Empresas_Segmento", ascending=False).reset_index(drop=True)

    cidades_html = gerar_tabela_html(cidades_df)

    print("==== DEBUG cidades_html ====")
    print(cidades_html)
    print("==== FIM DEBUG cidades_html ====")

    try:
        ticket = float(ticket_medio_str)
        ciclo = int(ciclo_vendas)
        novos_clientes = int(meta_clientes)
    except Exception as e:
        raise ValueError(f"Erro ao converter valores numericos: {str(e)}") from e

    conversao = consultar_taxa_conversao_openai(segmentos_normalizados)
    try:
        conversao = float(conversao)
        if conversao <= 0 or conversao > 1:
            raise ValueError("Conversão fora do intervalo aceitável.")
    except Exception as e:
        logger.warning(f"Conversão inválida detectada ({conversao}): {str(e)}. Usando valor padrão 0.2")
        conversao = 0.2

    if not isinstance(conversao, (float, int)) or conversao == 0:
        logger.warning(f"Conversão inválida detectada: {conversao}. Usando valor padrão 0.2")
        conversao = 0.2

    oportunidades = int(novos_clientes / conversao)
    prospeccoes = int(oportunidades / conversao)

    from datetime import datetime, timedelta
    receita_24_meses = 0
    hoje = datetime.today()
    for canal_index in range(20):
        dias_ate_primeira_venda = 90 + ciclo + (canal_index * 30)
        data_primeira_venda = hoje + timedelta(days=dias_ate_primeira_venda)
        meses_restantes = max(0, 24 - ((data_primeira_venda - hoje).days // 30))
        receita_24_meses += ticket * meses_restantes

    receita_mensal = ticket * novos_clientes * 20

    # Retorna diagnóstico estruturado com bloco das cidades em HTML incluído


    prompt_sem_cidades_html = f'''

🧠 DIAGNÓSTICO ESTRUTURADO – EMPRESA {empresa.upper()}

🔹 Local informado:
{origem}

🔹 Resumo sobre a empresa:
{resumo_empresa}

🔹 Produtos e Serviços:
{resumo_produto}

🔹 Perfil dos Clientes atendidos:
{resumo_clientes}

🔹 Dores e Benefícios:
{dores}

🔹 Modelo de Negócio:
{modelo_negocio}

🔹 Modelos de Parceria:
{conhecimento_modelos}

🔹 Perfis de Empresas Parceiras com FIT:
{perfis_e_modelos}

🔹 Serviços Agregados:
{servicos_agregados}

🔹 Necessidades de Prospecção:
- Meta Prevista: {novos_clientes} novo cliente/mês  
- Ciclo de Venda: {ciclo} dias  
- Índice de Conversão de Prospecção em Oportunidade: {int(taxa_pros * 100)}%  
- Número de Prospecções: {prospeccoes:,}  
- Índice de Conversão de Oportunidades em Vendas: {int(taxa_vendas * 100)}%  
- Número de Oportunidades: {oportunidades:,}

🔹 Retorno sobre o Investimento:
Ticket Médio: R${ticket:,.2f}
Receita Mensal com 20 canais: R${receita_mensal:,.2f}
Receita estimada em 24 meses: R${receita_24_meses:,.2f}

🔹 Dicas Estratégicas:

{dicas_estrategicas}

🔹 Chamada para Ação:
Sua empresa está pronta para crescer com uma estratégia sólida de canais de vendas.
Entre em contato com a AC Partners e comece agora o onboarding comercial com especialistas.
'''

    # RETORNE O PROMPT E O HTML DAS CIDADES SEPARADOS!

    return prompt_sem_cidades_html, cidades_html

def chamar_llm(prompt):
    """Chama a OpenAI para complementar o diagnóstico."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um consultor especialista em canais de vendas. "
                        "Siga exatamente a estrutura em 13 partes numeradas conforme o prompt. "
                        "Quando houver tabela HTML (como o bloco de Cidades com Potencial), mantenha o conteúdo original sem reescrever, resumir ou excluir. "
                        "Seu papel é complementar, não modificar a estrutura do prompt."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Erro na chamada a API OpenAI")
        raise RuntimeError("Erro na chamada a API OpenAI.") from e

# Funções auxiliares de detalhamento e conhecimento (sem alteração, já estavam bem formatadas)
def sugerir_cidades_openai(segmentos, total_necessario=30):
    try:
        prompt = (
            f"Liste {total_necessario} cidades brasileiras com potencial de mercado para empresas que atuam nos segmentos: {', '.join(segmentos)}. "
            "Considere demanda, perfil econômico, presença de agronegócio, indústria ou comércio local, dependendo do segmento. "
            "Forneça apenas os nomes das cidades, uma por linha."
        )
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um analista de inteligência de mercado."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        cidades = resposta.choices[0].message.content.strip().split("\n")
        return [c.strip().split("-")[0].strip() for c in cidades if c.strip()]
    except Exception as e:
        logger.error(f"Erro ao sugerir cidades com OpenAI: {str(e)}")
        return []

def detalhar_empresa_openai(site, nome_empresa, segmentos, clientes, produtos, modelo_negocio):
    """
    Gera um diagnóstico detalhado da empresa com base em todas as informações disponíveis.
    """
    try:
        prompt = (
            f"""
Empresa: {nome_empresa}
Site: {site}
Segmentos: {segmentos}
Clientes atendidos: {clientes}
Produtos vendidos: {produtos}
Como é vendido/modelo de negócio: {modelo_negocio}

Com base nesses dados, faça uma análise clara, objetiva e profissional sobre:
- O que a empresa faz, seu segmento e diferenciais.
- O potencial da empresa no mercado brasileiro.
- Como canais de venda podem ganhar dinheiro e crescer numa parceria com essa empresa (exemplifique).
- Destaque oportunidades, nichos, ou perfis de canal que podem performar melhor.
- Seja sintético, mas completo. Use no máximo 5 frases.
"""
        )
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Você é um consultor de canais B2B brasileiro, especialista em análise de potencial de empresas e oportunidades de parceria para canais."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao detalhar empresa: {str(e)}")
        return (
            f"Para mais informações, visite [{nome_empresa}]({site}). "
            f"Segmentos: {segmentos}. Clientes: {clientes}. Produtos: {produtos}."
        )

def detalhar_produto_openai(produto_texto, segmento):
    """
    Analisa profundamente o produto/solução, destacando ganhos para clientes nos segmentos informados,
    e mostra como canais de vendas podem gerar múltiplas receitas com esse produto.
    """
    try:
        prompt = (
            f"""
Produto/Solução: {produto_texto}
Segmento(s) de atuação: {segmento}

Com base nessas informações, faça uma análise detalhada e profissional:
- Explique a importância do produto para empresas dos segmentos informados, considerando desafios e oportunidades do mercado B2B brasileiro.
- Destaque todos os ganhos estratégicos, operacionais e financeiros que os clientes podem obter ao utilizá-lo.
- Aponte diferenciais competitivos do produto ou solução para o segmento.
- Analise como canais de vendas (revendas, consultorias, parceiros, etc.) podem gerar receitas adicionais com esse produto, incluindo modelos de comissão, serviços agregados, cross-sell, upsell, manutenção, suporte, projetos, integrações, treinamento, entre outros.
- Ofereça exemplos práticos de como um canal pode ampliar seu faturamento com esse produto em empresas desse segmento.
- Seja claro, objetivo e comercial, com até 5 frases bem desenvolvidas.
"""
        )
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um consultor de negócios B2B e especialista em geração de receita para canais de vendas no Brasil."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao detalhar produto: {str(e)}")
        return produto_texto

def detalhar_clientes_openai(clientes_lista, ticket_medio, segmentos, produtos, modelo_negocio):
    """
    Gera um diagnóstico completo do perfil de clientes da empresa, ganhos com o produto e oportunidades de receita para canais.
    """
    try:
        prompt = f"""
Empresa com os seguintes dados:
- Ticket médio: R${ticket_medio}
- Clientes atuais: {clientes_lista}
- Segmentos atendidos: {segmentos}
- Produtos/serviços vendidos: {produtos}
- Modelo de negócio: {modelo_negocio}

Com base nessas informações:
1. Analise o perfil dos clientes atuais e trace o perfil de cliente IDEAL para a empresa, considerando potencial de geração de valor e fit com produtos/serviços.
2. Explique todos os ganhos estratégicos e operacionais que os produtos/serviços podem trazer para esses clientes ideais.
3. Detalhe os tipos de receita que podem ser geradas para os canais de venda ao atuar com estes clientes e produtos (ex: comissão, implantação, suporte, treinamento, integração, serviços agregados, recorrência, etc).
4. Seja conciso, prático, comercial e objetivo. Use até 5 frases bem desenvolvidas.
"""
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Você é um consultor sênior de canais B2B com experiência em análise de mercado, perfis de clientes e oportunidades de geração de receita para parceiros."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao detalhar clientes: {str(e)}")
        return f"Clientes: {clientes_lista}. Ticket médio: R${ticket_medio}. Segmentos: {segmentos}. Produtos: {produtos}."

def consultar_taxas_software_b2b():
    prompt = (
        "No mercado brasileiro de software B2B:\n\n"
        "1. Qual é a taxa média de conversão de prospecções em oportunidades comerciais?\n"
        "2. Qual é a taxa média de conversão de oportunidades em vendas fechadas?\n\n"
        "Forneça os dois valores em porcentagem. Exemplo:\n"
        "- Prospecção para Oportunidade: 10%\n"
        "- Oportunidade para Venda: 20%\n"
    )
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em dados de mercado e vendas B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        texto = resposta.choices[0].message.content.strip()
        logger.info(f"Resposta OpenAI sobre conversão: {texto}")

        matches = re.findall(r"(\d{1,3})%", texto)
        if len(matches) >= 2:
            taxa_prospeccao = int(matches[0]) / 100
            taxa_vendas = int(matches[1]) / 100
            return taxa_prospeccao, taxa_vendas
        else:
            logger.warning("Não foi possível extrair as duas taxas. Usando padrão.")
            return 0.1, 0.2
    except Exception as e:
        logger.error(f"Erro ao consultar taxas de conversão: {str(e)}")
        return 0.1, 0.2

def buscar_modelos_parceria_empresa(
    nome_empresa,
    site,
    clientes,
    ticket_medio,
    ciclo_vendas,
    produto,
    segmentos,
    modelo_negocio,
    k=5
):
    """
    Busca no RAG e complementa com OpenAI os 5 melhores modelos de parceria para a empresa,
    detalhando o funcionamento e potencial de retorno de cada modelo.
    Usa cache para perguntas repetidas.
    """
    # Monte a pergunta personalizada e rica
    pergunta = (
        f"Considerando os seguintes dados da empresa:\n"
        f"- Nome: {nome_empresa}\n"
        f"- Site: {site}\n"
        f"- Segmentos: {segmentos}\n"
        f"- Clientes: {clientes}\n"
        f"- Produtos/Serviços: {produto}\n"
        f"- Modelo de negócio: {modelo_negocio}\n"
        f"- Ticket médio: R${ticket_medio}\n"
        f"- Ciclo médio de vendas: {ciclo_vendas} dias\n\n"
        f"Quais são os 5 melhores modelos de parceria para este negócio?\n"
        f"Para cada modelo, explique:\n"
        f"- Como funciona\n"
        f"- O que agrega para a empresa\n"
        f"- Como o parceiro pode obter ganhos (tipos de receita)\n"
        f"- Perfil ideal de parceiro para cada modelo\n"
        f"Responda de forma prática, clara e adaptada ao mercado brasileiro de canais."
    )

    # Cache para evitar repetição de busca
    cache_key = f"{pergunta}|k={k}"
    global _resposta_cache
    if "_resposta_cache" not in globals():
        _resposta_cache = {}
    if cache_key in _resposta_cache:
        return _resposta_cache[cache_key]

    # Busca no RAG (top k)
    resposta_rag = buscar_conhecimento(pergunta, k)

    # Prompt detalhado para a OpenAI complementar com base no que veio do RAG
    prompt = (
        f"Com base no conhecimento extraído a seguir, explique detalhadamente os 5 melhores modelos de parceria para a empresa informada, "
        f"destacando para cada um: funcionamento, ganhos para a empresa, ganhos para o parceiro (formas de receita), perfil ideal de canal e exemplos práticos.\n\n"
        f"Pergunta: {pergunta}\n\n"
        f"Conhecimento extraído do RAG:\n{resposta_rag}\n"
    )
    try:
        resposta_ia = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um consultor de canais de vendas B2B, especialista em modelagem de parcerias e geração de receita para canais no Brasil."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1000
        )
        resposta_final = (
            f"{resposta_rag}\n\n🔎 Complemento da IA:\n{resposta_ia.choices[0].message.content.strip()}"
        )
        _resposta_cache[cache_key] = resposta_final
        return resposta_final
    except Exception as e:
        logger.warning(f"Erro ao complementar conhecimento com OpenAI: {e}")
        _resposta_cache[cache_key] = resposta_rag
        return resposta_rag

def conhecimento_perfis(
    nome_empresa,
    site,
    clientes,
    ticket_medio,
    ciclo_vendas,
    produto,
    segmentos,
    modelo_negocio,
    k=5
):
    """
    Busca no RAG e complementa com OpenAI os 5 melhores modelos de parceria para a empresa
    e, para cada modelo, detalha o perfil ideal dos canais de vendas (estrutura, competências, segmento, etc).
    """
    pergunta = (
        f"Considerando os seguintes dados da empresa:\n"
        f"- Nome: {nome_empresa}\n"
        f"- Site: {site}\n"
        f"- Segmentos: {segmentos}\n"
        f"- Clientes: {clientes}\n"
        f"- Produtos/Serviços: {produto}\n"
        f"- Modelo de negócio: {modelo_negocio}\n"
        f"- Ticket médio: R${ticket_medio}\n"
        f"- Ciclo médio de vendas: {ciclo_vendas} dias\n\n"
        f"Quais são os 5 melhores modelos de parceria comercial para este negócio?\n"
        f"Para cada modelo, explique:\n"
        f"1. Nome do modelo\n"
        f"2. Descrição/função\n"
        f"3. Ganhos para a empresa\n"
        f"4. Ganhos para o parceiro (formas de receita)\n"
        f"5. PERFIL IDEAL DE CANAL/EMPRESA PARCEIRA para este modelo (estrutura, porte, competências, segmento, como atua, etc)\n"
        f"Se possível, dê exemplos práticos. Responda de forma estruturada e adaptada ao mercado brasileiro de canais."
    )

    # Cache para evitar repetição de busca
    cache_key = f"{pergunta}|k={k}"
    global _resposta_cache
    if "_resposta_cache" not in globals():
        _resposta_cache = {}
    if cache_key in _resposta_cache:
        return _resposta_cache[cache_key]

    resposta_rag = buscar_conhecimento(pergunta, k)

    prompt = (
        f"Com base no conhecimento extraído a seguir, explique detalhadamente os 5 melhores modelos de parceria para a empresa informada. "
        f"Para cada modelo, detalhe claramente:\n"
        f"1. Nome do modelo\n"
        f"2. Descrição/função\n"
        f"3. Ganhos para a empresa\n"
        f"4. Ganhos para o parceiro (formas de receita)\n"
        f"5. PERFIL IDEAL DE CANAL/EMPRESA PARCEIRA para este modelo (estrutura, porte, competências, segmento, como atua, etc)\n"
        f"6. Exemplos práticos\n\n"
        f"Pergunta: {pergunta}\n\n"
        f"Conhecimento extraído do RAG:\n{resposta_rag}\n"
    )
    try:
        resposta_ia = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um consultor sênior de canais de vendas B2B, especialista em modelagem de parcerias e perfis de canais no Brasil."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.45,
            max_tokens=1800
        )
        resposta_final = (
            f"{resposta_rag}\n\n🔎 Complemento da IA:\n{resposta_ia.choices[0].message.content.strip()}"
        )
        _resposta_cache[cache_key] = resposta_final
        return resposta_final
    except Exception as e:
        logger.warning(f"Erro ao complementar conhecimento com OpenAI: {e}")
        _resposta_cache[cache_key] = resposta_rag
        return resposta_rag

def relacao_servicos_agregados(
    nome_empresa,
    site,
    clientes,
    ticket_medio,
    ciclo_vendas,
    produto,
    segmentos,
    modelo_negocio,
    dores,
    beneficios,
    diferenciais,
    k=10
):
    """
    Gera uma análise dos 10 principais serviços agregados e fontes de receita
    que os canais de vendas podem oferecer em parceria com a empresa,
    considerando todas as características e diferenciais do negócio.
    """
    prompt = (
        f"Empresa: {nome_empresa}\n"
        f"Site: {site}\n"
        f"Clientes: {clientes}\n"
        f"Ticket médio: R${ticket_medio}\n"
        f"Ciclo médio de vendas: {ciclo_vendas} dias\n"
        f"Produtos/serviços: {produto}\n"
        f"Segmentos de atuação: {segmentos}\n"
        f"Modelo de negócio: {modelo_negocio}\n"
        f"Dores e benefícios para o cliente: {dores} | {beneficios}\n"
        f"Diferenciais competitivos: {diferenciais}\n\n"
        f"Com base nessas informações, liste e explique 10 serviços agregados diferentes e fontes de receita "
        f"que canais de vendas podem gerar e oferecer ao fechar parceria com essa empresa.\n"
        f"Para cada serviço, explique:\n"
        f"- O que é o serviço\n"
        f"- Como ele se conecta às dores e benefícios do cliente\n"
        f"- Como gera receita extra para o canal/parceiro\n"
        f"- Se possível, mencione diferenciais e exemplos práticos\n"
        f"Responda de forma prática, completa, adaptada ao mercado brasileiro e evite repetições."
    )

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um consultor de canais de vendas B2B, especialista em geração de receita e serviços agregados para parceiros no Brasil."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.45,
            max_tokens=1800
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao obter serviços agregados: {e}")
        return "Não foi possível gerar os serviços agregados neste momento."

def buscar_conhecimento_complementado(pergunta: str) -> str:
    """
    Consulta o RAG e complementa com a OpenAI para respostas mais completas.
    Garante resposta robusta para perguntas-chave do diagnóstico.
    """
    try:
        resposta_rag = buscar_conhecimento(pergunta, k=3)
        prompt = (
            "Você é um consultor de canais de vendas B2B. Responda à pergunta abaixo de forma clara, objetiva "
            "e baseada em boas práticas e exemplos do mercado brasileiro.\n\n"
            f"Pergunta: {pergunta}\n\n"
            f"A resposta deve ser complementar a este conteúdo já obtido:\n{resposta_rag}\n"
        )
        resposta_openai = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor de canais de vendas com experiência em parcerias B2B."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        return f"{resposta_rag}\n\n🔎 Complemento da IA:\n{resposta_openai.choices[0].message.content.strip()}"
    except Exception as e:
        logger.warning(f"Erro ao complementar conhecimento com OpenAI: {str(e)}")
        return buscar_conhecimento(pergunta)
    
def limpar_tabela_fake(diagnostico_final):
    # Procure o marcador do bloco real:
    marcador = " "
    partes = diagnostico_final.split(marcador)
    if len(partes) == 2:
        # Só retorna o texto até o marcador + o marcador + a tabela real depois
        texto_antes = partes[0].strip()
        bloco_real = marcador + partes[1]
        return f"{texto_antes}\n\n{bloco_real}"
    return diagnostico_final  # fallback se não encontrar

def titulos_html_sem_hash(texto):
    padrao = r"^###\s*([\d]+\. [^\n]+)"
    texto = re.sub(padrao, r"<strong>\1</strong>\n\n", texto, flags=re.MULTILINE)
    return texto

def gerar_dicas_estrategicas(
    site,
    segmentos,
    clientes,
    dores,
    beneficios,
    diferenciais,
    produtos,
    modelo_negocio,
    ticket_medio,
    ciclo_vendas,
    meta_clientes,
    modelos_parceria,
    perfis_canais,
    servicos_agregados
):
    """
    Gera dicas estratégicas personalizadas para aceleração de vendas, estruturação de canais e retenção de parceiros,
    usando todas as respostas do diagnóstico, modelos de parceria, perfis de canais e serviços agregados.
    """
    prompt = (
        f"Empresa com as seguintes características:\n"
        f"- Site: {site}\n"
        f"- Segmentos: {segmentos}\n"
        f"- Clientes: {clientes}\n"
        f"- Dores enfrentadas: {dores}\n"
        f"- Benefícios entregues: {beneficios}\n"
        f"- Diferenciais competitivos: {diferenciais}\n"
        f"- Produtos/serviços: {produtos}\n"
        f"- Modelo de negócio: {modelo_negocio}\n"
        f"- Ticket médio: R${ticket_medio}\n"
        f"- Ciclo médio de vendas: {ciclo_vendas} dias\n"
        f"- Meta de novos clientes por mês pelos canais: {meta_clientes}\n\n"
        f"Modelos de parceria considerados:\n{modelos_parceria}\n\n"
        f"Perfis de canais de vendas ideais:\n{perfis_canais}\n\n"
        f"Serviços agregados e fontes de receita possíveis:\n{servicos_agregados}\n\n"
        "Com base nesse contexto completo, liste 4 dicas estratégicas práticas, criativas e aplicadas ao mercado brasileiro para:\n"
        "- Acelerar as vendas via canais\n"
        "- Estruturar canais de vendas\n"
        "- Aumentar a retenção e engajamento dos parceiros\n"
        "As dicas devem ser detalhadas, profundas, realistas e voltadas ao contexto B2B. Evite repetir frases comuns, aprofunde de acordo com as informações fornecidas e relacione as dicas com os modelos de parceria, perfis de canais e serviços agregados apresentados."
    )
    try:
        resposta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um consultor sênior de canais de vendas B2B, especialista em aceleração comercial para empresas brasileiras."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.35
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Erro ao gerar dicas estratégicas: {e}")
        return (
            "- Inicie com 3 a 5 canais\n"
            "- Forneça treinamentos e acompanhamento semanal\n"
            "- Crie campanhas e ações de marketing\n"
            "- Corrija falhas com feedback dos canais"
        )