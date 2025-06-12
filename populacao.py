import pandas as pd
import plotly.express as px

def load_data():
    try:
        df = pd.read_csv('populacao_recife_2020.csv')
        df['data'] = pd.to_datetime(df['data'])
        return df
    except FileNotFoundError:
        return None

# Função para gerar tabela, gráfico e estatísticas
def generate_interface(plot_type):
    df = load_data()
    if df is None:
        return "Erro: populacao_recife_2020.csv não encontrado.", None, ""

    # Tabela
    table = df[['data', 'Ano', 'Mes', 'Populacao', 'Taxa_Crescimento']].to_html(index=False)

    # Gráfico
    if plot_type == "População":
        fig = px.line(df, x='data', y='Populacao', title='População Estimada Mensal em Recife (2020)',
                    labels={'Populacao': 'População', 'data': 'Data'})
    else:
        fig = px.line(df, x='data', y='Taxa_Crescimento', title='Taxa de Crescimento Populacional Mensal em Recife (2020)',
                    labels={'Taxa_Crescimento': 'Taxa de Crescimento (%)', 'data': 'Data'})

    # Estatísticas
    pop_mean = df['Populacao'].mean()
    growth_mean = df['Taxa_Crescimento'].mean()
    pop_change = df['Populacao'].iloc[-1] - df['Populacao'].iloc[0]
    stats = f"""
    **Estatísticas:**
    - População média: {pop_mean:,.0f} habitantes  
    - Taxa de crescimento média mensal: {growth_mean:.4f}%  
    - Variação total da população em 2020: {pop_change:,.0f} habitantes
    """

    return table, fig, stats