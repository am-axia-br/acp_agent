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

    # Converte quebras de linha em parágrafos
    resposta_formatada = "".join(f"<p>{linha.strip()}</p>" for linha in resposta.split("\n") if linha.strip())

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

