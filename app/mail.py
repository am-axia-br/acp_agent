import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def enviar_email(data, resposta):
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

    resposta_formatada = resposta.replace('. ', '.<br><br>')

    corpo = f"""
    <html>
    <head>
      <style>
        body {{
          font-family: 'Inter', sans-serif;
          color: #333;
          line-height: 1.6;
        }}
        .container {{
          max-width: 700px;
          margin: auto;
          padding: 30px;
          background-color: #fff9f5;
          border: 1px solid #ffe1d6;
          border-radius: 10px;
        }}
        .titulo {{
          font-size: 20px;
          font-weight: bold;
          color: #a638ec;
          margin-bottom: 15px;
        }}
        .paragrafo {{
          margin-bottom: 12px;
        }}
        .assinatura {{
          margin-top: 30px;
          font-size: 14px;
          color: #999;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="titulo">Resumo do Diagnóstico Comercial</div>

        <div class="paragrafo">
          <strong>Olá, {data['nome']},</strong><br><br>
          Abaixo está o diagnóstico detalhado com base nas suas respostas:
        </div>

        <div class="paragrafo">{resposta_formatada}</div>

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
            server.sendmail(remetente, destinatario, msg.as_string())
        print("✅ E-mail enviado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao enviar e-mail: {e}")
