import pandas as pd

# Carregar base de dados do IBGE (arquivo deve estar no mesmo diretório ou especificar caminho completo)
df = pd.read_excel("Tabela 14.xlsx")

# Renomear colunas para facilitar o uso
df.columns = [
    "Municipio", "Codigo_CNAE", "Descricao_CNAE", "Unidades_Locais",
    "Pessoal_Total", "Pessoal_Assalariado", "Assalariado_Medio",
    "Remuneracao_Mil_R$", "Salario_Medio_SM", "Salario_Medio_R$"
]

# Limpar dados irrelevantes e manter apenas registros válidos
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
    filtrado = df[df["Descricao_CNAE"].astype(str).str.contains(segmento, case=False, na=False)]
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

def gerar_tabela_html(dataframe):
    """
    Gera uma tabela HTML estilizada com os dados de municípios recomendados
    """
    linhas = ""
    for _, row in dataframe.iterrows():
        linhas += f"<tr><td>{row['Municipio']}</td><td>{int(row['Unidades_Locais'])}</td><td>R${row['Salario_Medio_R$']:.2f}</td></tr>"

    return f"""
    <div class='paragrafo'>
    <h3 style='color:#5e17eb;'>\ud83d\udccd Top Municípios com Maior Potencial para Canais</h3>
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
