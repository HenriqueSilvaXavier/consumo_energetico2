import matplotlib.pyplot as plt
import pandas as pd
import autenticar

df_previsao, df = autenticar.carregar_e_processar_dados()
colunas_numericas = df.drop(columns=["Unnamed: 19", "hora", "mes", 'datetime', 'Data', 'Hora UTC']).columns.tolist()

# Função para gerar o gráfico meteorológico
def gerar_grafico_meteorologico(variaveis, tipo_grafico, agregacao, titulo, grid):
    
    # Carregar os dados

    # Checando se há variáveis selecionadas
    if not variaveis:
        return None
    
    # Processamento de dados baseado no tipo de agregação
    if agregacao == "Hora":
        df_agg = df[["Data", "Hora UTC"] + variaveis].copy()
        df_agg["DataHora"] = pd.to_datetime(df_agg["Data"].astype(str) + " " + df_agg["Hora UTC"], errors='coerce')
        dados = df_agg.dropna(subset=["DataHora"] + variaveis)
        dados["Hora"] = dados["DataHora"].dt.hour

        # Converte as variáveis para numérico (com erro coercitivo)
        for var in variaveis:
            dados[var] = pd.to_numeric(dados[var], errors='coerce')

        # Agrupa pela nova coluna 'Hora'
        media = dados.groupby("Hora")[variaveis].mean()
        eixo_x = media.index
        xlabel = "Hora do Dia (UTC)"
    elif agregacao == "Dia":
        dados = df[["Data"] + variaveis].dropna()
        media = dados.groupby("Data")[variaveis].mean()
        eixo_x = media.index
        xlabel = "Data"
    else:  # Mês
        df["Data"] = pd.to_datetime(df["Data"], errors='coerce')  # ADICIONE ESTA LINHA
        dados = df[["Data"] + variaveis].dropna()
        media = dados.groupby(dados["Data"].dt.to_period("M"))[variaveis].mean()
        media.index = media.index.to_timestamp()
        eixo_x = media.index
        xlabel = "Mês"

    # Criando o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    if tipo_grafico == "Linha":
        for var in variaveis:
            ax.plot(eixo_x, media[var], marker="o", label=var)
    elif tipo_grafico == "Barras":
        largura = 0.8 / len(variaveis)
        for i, var in enumerate(variaveis):
            deslocamento = pd.to_timedelta(i * largura, unit='D') - pd.to_timedelta(largura * len(variaveis) / 2, unit='D')
            ax.bar(eixo_x + deslocamento, media[var], width=largura, label=var)
    else:
        for var in variaveis:
            ax.fill_between(eixo_x, media[var], alpha=0.5, label=var)

    ax.set_title(titulo if titulo.strip() else "Gráfico Meteorológico")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Valor")
    ax.grid(grid)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
