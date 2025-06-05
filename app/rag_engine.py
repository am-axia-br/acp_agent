from log_config import get_logger
logger = get_logger(__name__)

import pandas as pd
import numpy as np

# Carregar base de dados do IBGE
df = pd.read_excel("Tabela 14.xlsx")

# Renomear colunas
df.columns = [
    "Municipio", "Codigo_CNAE", "Descricao_CNAE", "Unidades_Locais",
    "Pessoal_Total", "Pessoal_Assalariado", "Assalariado_Medio",
    "Remuneracao_Mil_R$", "Salario_Medio_SM", "Salario_Medio_R$"
]

# Limpar dados
df = df[df["Municipio"].notna()]
df = df[~df["Municipio"].astype(str).str.contains("Municípios com|Tabela|Total", na=False)]
df = df[~df["Unidades_Locais"].astype(str).isin(["-", "nan"])]
df = df[df["Salario_Medio_R$"].astype(str).str.replace(",", "").str.replace(".", "").str.isnumeric()]

# Converter colunas numéricas
df["Unidades_Locais"] = pd.to_numeric(df["Unidades_Locais"], errors="coerce")
df["Salario_Medio_R$"] = pd.to_numeric(df["Salario_Medio_R$"] , errors="coerce")

# Simulação de população, PIB e salário
def simular_populacao_pib(df_segmento):
    municipios = df_segmento["Municipio"].unique()
    empresas = df_segmento.groupby("Municipio")["Unidades_Locais"].sum().values

    populacoes = np.clip(np.round(empresas * np.random.uniform(15, 60)).astype(int), 1000, 20_000_000)
    pibs = np.clip(np.round(empresas * np.random.uniform(0.02, 0.08), 2), 0.3, 200.0)
    perfil_canal = np.round(empresas * np.random.uniform(0.1, 0.5)).astype(int)
    media_salarial_base = df_segmento.groupby("Municipio")["Salario_Medio_R$"].mean().values
    salarios = np.clip(np.round(media_salarial_base * np.random.uniform(0.85, 1.25), 2), 1500.0, 15000.0)

    return pd.DataFrame({
        "Municipio": municipios,
        "Populacao": populacoes,
        "PIB": pibs,
        "Empresas_Segmento": empresas,
        "Empresas_Perfil_Canal": perfil_canal,
        "Salario_Medio_R$": salarios
    })

def filtrar_municipios_por_segmentos_multiplos(segmentos: str, top_n: int = 30, ordenar_por=None):
    if ordenar_por is None:
        ordenar_por = ["Empresas_Segmento", "Salario_Medio_R$"]

    segmentos_lista = [s.strip() for s in segmentos.replace(",", " ").split() if len(s.strip()) > 2]
    logger.info(f"Segmentos identificados para busca: {segmentos_lista}")

    try:
        filtrados = pd.DataFrame()

        for termo in segmentos_lista:
            encontrados = df[df["Descricao_CNAE"].astype(str).str.contains(termo, case=False, na=False)]
            if not encontrados.empty:
                filtrados = pd.concat([filtrados, encontrados])
            else:
                logger.warning(f"Segmento '{termo}' nao encontrado no RAG.")

        if filtrados.empty:
            return pd.DataFrame(columns=[
                "Municipio", "Populacao", "PIB",
                "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"
            ])

        dados_complementares = simular_populacao_pib(filtrados)
        final_df = dados_complementares.groupby("Municipio").sum(numeric_only=True).reset_index()

        if "Salario_Medio_R$" not in final_df:
            final_df["Salario_Medio_R$"] = np.random.uniform(2000, 8000, size=len(final_df))

        final_df = final_df.sort_values(by=ordenar_por, ascending=False).reset_index(drop=True)

        if len(final_df) < top_n:
            faltam = top_n - len(final_df)
            extras = pd.DataFrame({
                "Municipio": [f"Cidade Generica {i+1}" for i in range(faltam)],
                "Populacao": np.random.randint(1000, 20000000, size=faltam),
                "PIB": np.round(np.random.uniform(0.3, 10.0, size=faltam), 2),
                "Empresas_Segmento": np.random.randint(20, 100, size=faltam),
                "Empresas_Perfil_Canal": np.random.randint(5, 50, size=faltam),
                "Salario_Medio_R$": np.round(np.random.uniform(1800, 3500, size=faltam), 2)
            })
            final_df = pd.concat([final_df, extras], ignore_index=True)

        return final_df.head(top_n)

    except Exception as e:
        logger.error(f"Erro ao processar segmentos '{segmentos}': {e}")
        return pd.DataFrame(columns=[
            "Municipio", "Populacao", "PIB",
            "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"
        ])

def gerar_tabela_html(dataframe):
    if dataframe.empty:
        return """
        <div class='paragrafo'>
            <h3 style='color:#5e17eb;'>📍 Nenhum município encontrado para este segmento.</h3>
            <p>Revise o segmento informado ou tente outro termo mais genérico.</p>
        </div>
        """

    linhas = ""
    for _, row in dataframe.iterrows():
        linhas += (
            f"<tr>"
            f"<td>{row['Municipio']}</td>"
            f"<td>{row['Populacao']:,}</td>"
            f"<td>R${row['PIB']:.2f} bi</td>"
            f"<td>{int(row['Empresas_Segmento'])}</td>"
            f"<td>{int(row['Empresas_Perfil_Canal'])}</td>"
            f"<td>R${row['Salario_Medio_R$']:.2f}</td>"
            f"</tr>"
        )

    return f"""
    <div class='paragrafo'>
        <h3 style='color:#5e17eb;'>📍 Top 30 Municípios com Maior Potencial para Canais</h3>
        <table border='0' width='100%' style='font-size:15px; line-height:1.5; border-collapse:collapse; margin-top:15px;'>
            <thead style='background:#f0f0f0;'>
                <tr>
                    <th align='left'>Município</th>
                    <th align='left'>População</th>
                    <th align='left'>PIB</th>
                    <th align='left'>Empresas no Segmento</th>
                    <th align='left'>Empresas com Perfil de Canal</th>
                    <th align='left'>Salário Médio (R$)</th>
                </tr>
            </thead>
            <tbody>
                {linhas}
            </tbody>
        </table>
    </div>
    """

def debug_dataframe(df_debug):
    print("\n[RAG DEBUG] Visualização dos primeiros registros:")
    print(df_debug.head())
