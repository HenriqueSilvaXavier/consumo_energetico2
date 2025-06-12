import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from PIL import Image
import io

def load_prepare_sazonalidade():
    df_sazonalidade = pd.read_csv('CONSUMO_energia.csv', sep=';', header=5)
    meses = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']

    def preparar_ano(df, ano, regiao='Nordeste'):
        linha = df[df['Unnamed: 0'] == regiao].iloc[0]
        valores = linha[meses].values

        df_ano = pd.DataFrame({
            'Mes': meses,
            'Consumo_MWh': valores
        })

        df_ano['Consumo_MWh'] = df_ano['Consumo_MWh'].replace('', pd.NA)
        df_ano['Consumo_MWh'] = df_ano['Consumo_MWh'].str.replace('.', '', regex=False)
        df_ano['Consumo_MWh'] = pd.to_numeric(df_ano['Consumo_MWh'], errors='coerce')

        meses_map = {m: i+1 for i, m in enumerate(meses)}
        df_ano['Mes_num'] = df_ano['Mes'].map(meses_map)
        df_ano['Data'] = pd.to_datetime({'year': ano, 'month': df_ano['Mes_num'], 'day': 1})
        df_ano = df_ano.set_index('Data')

        return df_ano[['Consumo_MWh']]

    df_2020 = preparar_ano(df_sazonalidade, 2020)
    df_2021 = preparar_ano(df_sazonalidade, 2021)
    df_ne_2anos = pd.concat([df_2020, df_2021])
    df_ne_2anos['Consumo_MWh'] = df_ne_2anos['Consumo_MWh'].interpolate(method='linear')

    return df_ne_2anos

def gerar_decomposicao():
    df = load_prepare_sazonalidade()
    result = seasonal_decompose(df['Consumo_MWh'], model='additive', period=12)
    fig = result.plot()
    fig.set_size_inches(10, 6)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)  # ✅ Corrigido: retorna imagem PIL

def gerar_grafico_barra():
    df = load_prepare_sazonalidade()
    result = seasonal_decompose(df['Consumo_MWh'], model='additive', period=12)
    seasonal_monthly = result.seasonal.groupby(result.seasonal.index.month).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(seasonal_monthly.index, seasonal_monthly, color='green')
    ax.set_title('Padrão Sazonal Médio Mensal do Consumo de Energia - Nordeste')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Sazonalidade Média (Consumo MWh)')
    ax.set_xticks(seasonal_monthly.index)
    ax.set_xticklabels(['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN',
                        'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ'])
    ax.grid(axis='y')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)  # ✅ Corrigido: retorna imagem PIL

def gerar_grafico_estacoes():
    df = load_prepare_sazonalidade()
    result = seasonal_decompose(df['Consumo_MWh'], model='additive', period=12)

    df_sazonal = result.seasonal.reset_index()
    df_sazonal.columns = ['Data', 'Sazonalidade']
    df_sazonal['Ano'] = df_sazonal['Data'].dt.year
    df_sazonal['Mês'] = df_sazonal['Data'].dt.month

    def mes_para_estacao(mes):
        if mes in [12, 1, 2]:
            return 'Verão'
        elif mes in [3, 4, 5]:
            return 'Outono'
        elif mes in [6, 7, 8]:
            return 'Inverno'
        else:
            return 'Primavera'

    df_sazonal['Estação'] = df_sazonal['Mês'].apply(mes_para_estacao)
    df_pivot = df_sazonal.groupby(['Ano', 'Estação'])['Sazonalidade'].mean().unstack()
    df_pivot = df_pivot[['Verão', 'Outono', 'Inverno', 'Primavera']]

    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title('Sazonalidade Média Empilhada por Estação e Ano - Nordeste')
    ax.set_ylabel('Sazonalidade Média (Consumo MWh)')
    ax.set_xlabel('Ano')
    ax.legend(title='Estação')
    ax.grid(axis='y')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)  # ✅ Retorna imagem PIL


def mostrar_grafico(escolha):
    if escolha == "Decomposição Sazonal":
        return gerar_decomposicao()
    elif escolha == "Padrão Sazonal Médio Mensal":
        return gerar_grafico_barra()
    else:
        return gerar_grafico_estacoes()
