import pandas as pd

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

def filtrar_municipios_por_segmento(segmento: str, top_n: int = 20):
    """
    Filtra os municípios com maior potencial para o segmento informado,
    considerando quantidade de unidades locais e salário médio.
    """
    try:
        filtrado = df[df["Descricao_CNAE"].astype(str).str.contains(segmento, case=False, na=False)]
        if filtrado.empty:
            return pd.DataFrame(columns=["Municipio", "Unidades_Locais", "Salario_Medio_R$"])

        agrupado = (
            filtrado.groupby("Municipio")
            .agg({
                "Unidades_Locais": "sum",
                "Salario_Medio_R$": "mean"
            })
            .sort_values(by=["Unidades_Locais", "Salario_Medio_R$"], ascending=False)
            .reset_index()
        )
        return agrupado.head(top_n)

    except Exception as e:
        print(f"[RAG ENGINE] Erro ao filtrar segmento '{segmento}': {e}")
        return pd.DataFrame(columns=["Municipio", "Unidades_Locais", "Salario_Medio_R$"])

def gerar_tabela_html(dataframe):
    """
    Gera uma tabela HTML estilizada com os dados de municípios recomendados
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
        linhas += f"<tr><td>{row['Municipio']}</td><td>{int(row['Unidades_Locais'])}</td><td>R${row['Salario_Medio_R$']:.2f}</td></tr>"

    return f"""
    <div class='paragrafo'>
    <h3 style='color:#5e17eb;'>📍 Top Municípios com Maior Potencial para Canais</h3>
    <table border='0' width='100%' style='font-size:15px; line-height:1.5; border-collapse:collapse;'>
        <tr style='background:#f0f0f0;'>
            <th align='left'>Município</th>
            <th align='left'>Unidades Locais</th>
            <th align='left'>Salário Médio (R$)</th>
        </tr>
        {linhas}
    </table>
    </div>
    """
