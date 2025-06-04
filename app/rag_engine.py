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
df["Salario_Medio_R$"] = pd.to_numeric(df["Salario_Medio_R$"], errors="coerce")

# Simulação de população e PIB
def simular_populacao_pib(df_segmento):
    municipios = df_segmento["Municipio"].unique()
    populacoes = np.random.randint(30000, 1500000, size=len(municipios))
    pibs = np.round(np.random.uniform(0.5, 50.0, size=len(municipios)), 2)
    perfil_canal = np.round(df_segmento.groupby("Municipio")["Unidades_Locais"].sum() * np.random.uniform(0.1, 0.5)).astype(int).values
    return pd.DataFrame({
        "Municipio": municipios,
        "Populacao": populacoes,
        "PIB": pibs,
        "Empresas_Segmento": df_segmento.groupby("Municipio")["Unidades_Locais"].sum().values,
        "Empresas_Perfil_Canal": perfil_canal
    })

def filtrar_municipios_por_segmento(segmento: str, top_n: int = 30, ordenar_por=None):
    """
    Filtra os municípios com maior potencial para o segmento informado.
    Permite definir a ordem de classificação com o parâmetro ordenar_por.
    """
    if ordenar_por is None:
        ordenar_por = ["Empresas_Segmento", "Salario_Medio_R$"]

    try:
        filtrado = df[df["Descricao_CNAE"].astype(str).str.contains(segmento, case=False, na=False)]

        if filtrado.empty:
            return pd.DataFrame(columns=[
                "Municipio", "Populacao", "PIB",
                "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"
            ])

        dados_complementares = simular_populacao_pib(filtrado)
        salario_medio = filtrado.groupby("Municipio")["Salario_Medio_R$"].mean().reset_index()
        final_df = dados_complementares.merge(salario_medio, on="Municipio")
        final_df = final_df.sort_values(by=ordenar_por, ascending=False).reset_index(drop=True)

        return final_df.head(top_n)

    except Exception as e:
        print(f"[RAG ENGINE] Erro ao filtrar segmento '{segmento}': {e}")
        return pd.DataFrame(columns=[
            "Municipio", "Populacao", "PIB",
            "Empresas_Segmento", "Empresas_Perfil_Canal", "Salario_Medio_R$"
        ])

def gerar_tabela_html(dataframe):
    """
    Gera uma tabela HTML estilizada com os dados de municípios recomendados.
    """
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
    <table border='0' width='100%' style='font-size:15px; line-height:1.5; border-collapse:collapse;'>
        <tr style='background:#f0f0f0;'>
            <th align='left'>Município</th>
            <th align='left'>População</th>
            <th align='left'>PIB</th>
            <th align='left'>Empresas no Segmento</th>
            <th align='left'>Empresas com Perfil de Canal</th>
            <th align='left'>Salário Médio (R$)</th>
        </tr>
        {linhas}
    </table>
    </div>
    """
