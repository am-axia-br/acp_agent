import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import re

load_dotenv()

def formatar_tabela_cidades(texto):
    cidades = []
    atual = {}

    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha:
            continue

        if re.match(r"^\d{1,2}\.\s", linha):
            if atual:
                cidades.append(atual)
            atual = {"nome": linha}
        elif linha.lower().startswith("população"):
            atual["populacao"] = linha
        elif linha.lower().startswith("pib"):
            atual["pib"] = linha
        elif "segmentos" in linha.lower():
            atual["segmentos"] = linha
        elif "principal" in linha.lower():
            atual["principal"] = linha
        elif "perfil para parceria" in linha.lower():
            atual["perfil"] = linha

    if atual:
        cidades.append(atual)

    if not cidades:
        return None

    linhas_html = ""
    for c in cidades:
        linhas_html += f"""
        <tr><td colspan="2" style="font-weight:bold; font-size: 16px; padding-top: 15px;">{c.get("nome", "")}</td></tr>
        <tr><td>📍 {c.get("populacao", "")}</td><td>{c.get("pib", "")}</td></tr>
        <tr><td>{c.get("segmentos", "")}</td><td>{c.get("principal", "")}</td></tr>
        <tr><td colspan="2" style="padding-bottom:10px;">{c.get("perfil", "")}</td></tr>
        <tr><td colspan="2"><hr></td></tr>
        """

    return f"""
    <div class="paragrafo">
      <h3 style="color: #5e17eb;">📊 Cidades com maior potencial para parcerias</h3>
      <table border="0" width="100%" style="font-size:15px; line-height:1.5; border-collapse:collapse;">
        {linhas_html}
      </table>
    </div>
    """

def enviar_email(data, resposta, copia_para=None):
    remetente = os.getenv("EMAIL_REMETENTE")
    senha = os.getenv("EMAIL_SENHA")
    destinatario = data.get("email") or os.getenv("EMAIL_DESTINO")

    if not all([remetente, senha, destinatario]):
        print("❌ Falta configurar EMAIL_REMETENTE, EMAIL_SENHA ou destinatário.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Diagnóstico de Canais - {data['empresa']}"
    msg["From"] = remetente
    msg["To"] = destinatario

    if copia_para:
        msg["Bcc"] = ", ".join(copia_para)

    # Separa o trecho das cidades
    trecho_cidades = formatar_tabela_cidades(resposta)
    if trecho_cidades:
        resposta = re.sub(r"\d{1,2}\..+?(?:perfil para parceria.+?\n?)", "", resposta, flags=re.DOTALL | re.IGNORECASE)

    # Remove marcadores indesejados (** e #)
    resposta = re.sub(r"\*\*(.*?)\*\*", r"\1", resposta)
    resposta = re.sub(r"#*", "", resposta)

    # Formata os parágrafos do restante da resposta
    resposta_formatada = "".join(
        f"<p style='margin-bottom: 15px;'>{linha.strip()}</p>"
        for linha in resposta.split("\n") if linha.strip()
    )

    corpo = f"""
    <html>
    <head>
      <style>
        body {{
          font-family: 'Inter', sans-serif;
          color: #333;
          background-color: #fdfbfb;
          padding: 20px;
        }}
        .container {{
          max-width: 700px;
          margin: auto;
          background-color: #ffffff;
          border: 1px solid #e0dede;
          border-radius: 10px;
          padding: 30px;
          box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }}
        .titulo {{
          font-size: 20px;
          font-weight: bold;
          color: #7b2cbf;
          margin-bottom: 20px;
        }}
        .paragrafo {{
          font-size: 16px;
          margin-bottom: 20px;
          line-height: 1.6;
        }}
        .assinatura {{
          margin-top: 40px;
          font-size: 14px;
          color: #999;
          text-align: center;
        }}
        table td {{
          padding: 4px 8px;
          vertical-align: top;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="titulo">Resumo do Diagnóstico Comercial</div>

        <div class="paragrafo">
          <strong>Olá, {data['nome']},</strong><br><br>
          Com base nas suas respostas, desenvolvemos abaixo o diagnóstico detalhado para sua empresa:
        </div>

        {resposta_formatada}

        {trecho_cidades or ""}

        <div class="assinatura">
          Enviado automaticamente por IA – Agente ACP<br>
          © AC Partners – Conectar Expande!
        </div>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(corpo, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(remetente, senha)
            todos_destinatarios = [destinatario]
            if copia_para:
                todos_destinatarios.extend(copia_para)
            server.sendmail(remetente, todos_destinatarios, msg.as_string())
        print("✅ E-mail enviado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao enviar e-mail: {e}")
