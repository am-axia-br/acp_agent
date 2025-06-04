# log_config.py
import logging
import os

# Criação do diretório de logs, se não existir
os.makedirs("logs", exist_ok=True)

# Configuração global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Função auxiliar para criar logger por módulo
def get_logger(name):
    return logging.getLogger(name)
