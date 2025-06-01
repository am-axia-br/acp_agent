import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def enviar_email(data, resposta):
    remetente = os.getenv("EMAIL_REMETENTE")
    senha = os.getenv("EMAIL_SENHA")
    destinatario = os.getenv("EMAIL_DESTINO", data["email"])

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Diagnóstico de Canais - {data['empresa']}"
    msg["From"] = remetente
    msg["To"] = destinatario

    corpo = f"""
    <h3>Olá, {data['nome']}</h3>
    <p>Segue abaixo o diagnóstico com base nas suas respostas:</p>
    <p><pre>{resposta}</pre></p>
    <p>Atenciosamente,<br>IA AcPartners</p>
    """

    msg.attach(MIMEText(corpo, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(remetente, senha)
            server.sendmail(remetente, destinatario, msg.as_string())
        print("E-mail enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")
