import gradio as gr
import json
import bcrypt
import pyotp
import qrcode
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import holidays
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import io

USERS_DB = "users.json"

def load_users():
    try:
        with open(USERS_DB, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USERS_DB, "w") as f:
        json.dump(users, f, indent=2)

def create_qr_code(uri):
    qr = qrcode.make(uri)
    buf = BytesIO()
    qr.save(buf)
    buf.seek(0)
    return Image.open(buf)

def register(email, password):
    users = load_users()
    if email in users:
        return "Email já cadastrado.", None
    mfa_secret = pyotp.random_base32()
    uri = pyotp.totp.TOTP(mfa_secret).provisioning_uri(name=email, issuer_name="GradioApp")
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[email] = {"password": hashed_pw, "mfa_secret": mfa_secret, "verified": False}
    save_users(users)
    return "Usuário criado! Escaneie o QR Code com Google Authenticator.", create_qr_code(uri)

def autenticar(email, senha):
    users = load_users()
    user = users.get(email)
    if not user:
        return "Usuário não encontrado.", False, "", False
    if not bcrypt.checkpw(senha.encode(), user["password"].encode()):
        return "Senha incorreta.", False, "", False
    return "Digite o token MFA", False, email, True

def verificar_mfa(email, token):
    users = load_users()
    user = users.get(email)
    if not user:
        return "Usuário não encontrado.", False
    totp = pyotp.TOTP(user["mfa_secret"])
    if totp.verify(token):
        user["verified"] = True
        save_users(users)
        return "MFA verificado com sucesso! Faça login novamente.", True
    return "Token inválido.", False

def carregar_e_processar_dados():
    try:
        df = pd.read_csv(
            "INMET_NE_PE_A301_RECIFE_01-01-2020_A_31-12-2020.CSV",
            encoding='latin1',
            sep=';',
            skiprows=8
        )
        df.columns = [col.strip() for col in df.columns]
        numeric_cols = ["PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)", "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        df["datetime"] = pd.to_datetime(df["Data"] + " " + df["Hora UTC"], errors='coerce')
        df["hora"] = df["datetime"].dt.hour
        df["mes"] = df["datetime"].dt.month
        df = df[df["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"].notna()]
        df_previsao = df[["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)", "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)", "hora", "mes"]].dropna().copy()
        df_previsao = df.astype({
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'float64',
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'float64',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'float64',
            'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'float64',
            'hora': 'int64',
            'mes': 'int64'
        })
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        return df_previsao, df
    except Exception:
        np.random.seed(42)
        n_samples = 1000
        return pd.DataFrame({
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": np.random.normal(25, 5, n_samples),
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': np.random.exponential(2, n_samples),
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': np.random.normal(1013, 10, n_samples),
            'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': np.random.uniform(30, 90, n_samples),
            'hora': np.random.randint(0, 24, n_samples),
            'mes': np.random.randint(1, 13, n_samples)
        })

def treinar_modelo(df):
    feature_cols = ["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)", "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)", "hora", "mes"]
    X = df[feature_cols]
    y = df["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

df_previsao, df = carregar_e_processar_dados()
modelo = treinar_modelo(df_previsao)

def prever_temperatura(precipitacao, pressao, umidade, hora, mes):
    entrada = pd.DataFrame([{
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": precipitacao,
        "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": pressao,
        "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": umidade,
        "hora": hora,
        "mes": mes
    }])
    pred = modelo.predict(entrada)[0]
    df_temp = df.copy()
    df_temp = pd.concat([df_temp, entrada.assign(temperatura=pred)], ignore_index=True)
    fig1, fig2, fig3 = gerar_graficos(modelo, df_temp)
    return round(pred, 2), fig1, fig2, fig3

def gerar_graficos(modelo, df):
    feature_cols = ["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)", "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)", "hora", "mes"]
    X = df[feature_cols]
    y = df["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"]
    y_pred = modelo.predict(X)
    fig1, ax1 = plt.subplots()
    ax1.scatter(y, y_pred, alpha=0.5)
    ax1.set_xlabel("Temperatura real")
    ax1.set_ylabel("Temperatura prevista")
    ax1.set_title("Previsão de Temperatura")
    fig2, ax2 = plt.subplots()
    ax2.barh(feature_cols, modelo.feature_importances_, color="green")
    ax2.set_title("Importância das Variáveis")
    fig3, ax3 = plt.subplots()
    ax3.hist(y - y_pred, bins=30, color="orange", edgecolor="black")
    ax3.set_title("Distribuição dos Erros")
    ax3.set_xlabel("Erro")
    return fig1, fig2, fig3

# === Interface com abas ===
with gr.Blocks() as app:
    estado_login = gr.State(False)
    email_login = gr.State("")
    mostrar_mfa = gr.State(False)

    def atualiza_mfa_visibilidade(mfa_visivel, email):
        return (
            gr.update(visible=mfa_visivel, value=email),
            gr.update(visible=mfa_visivel),
            gr.update(visible=mfa_visivel),
            gr.update(visible=mfa_visivel)
        )

    def atualizar_visibilidade_grupos(autenticado):
        return (
            gr.update(visible=autenticado),
            gr.update(visible=autenticado),
            gr.update(visible=autenticado),
            gr.update(visible=autenticado),
            gr.update(visible=autenticado),
            gr.update(visible=autenticado)
        )
    
    df_feriados = pd.read_csv('CARGA_ENERGIA_2020.csv', sep=';')
    df_feriados['din_instante'] = pd.to_datetime(df_feriados['din_instante'])

    # Coluna de feriados
    br_holidays = holidays.Brazil(years=2020)
    df_feriados['feriado'] = df_feriados['din_instante'].dt.date.isin(br_holidays)

    subsistemas = df_feriados['nom_subsistema'].unique().tolist()

    df_feriados["data"] = pd.to_datetime(df_feriados["din_instante"])
    df_feriados["dia_semana"] = df_feriados["data"].dt.dayofweek + 1  # pandas: 0=Seg, 6=Dom -> ajusta para 1=Dom, 7=Sáb
    # Ajustar para 1=Domingo, ..., 7=Sábado: 
    # dayofweek pandas: Monday=0, Sunday=6
    # Quer 1=Domingo, ..., 7=Sábado. Domingo = 7 no padrão pandas. Vamos mapear:
    df_feriados["dia_semana"] = df_feriados["data"].dt.weekday.apply(lambda x: (x + 2) % 7 + 1)
    df_feriados["mes"] = df_feriados["data"].dt.month

    # Codifica a região como label numérico
    le = LabelEncoder()
    df_feriados["label"] = le.fit_transform(df_feriados["nom_subsistema"])

    # Features e target
    feature_cols = ["val_cargaenergiamwmed", "dia_semana", "mes"]
    X = df_feriados[feature_cols]
    y = df_feriados["label"]

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina modelo Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    with gr.Tabs() as abas:
        print(df.columns)

        def prever_regiao(carga, dia_semana, mes):
            input_data = pd.DataFrame([[carga, dia_semana, mes]], columns=feature_cols)
            pred_label = rf.predict(input_data)[0]
            regiao = le.inverse_transform([pred_label])[0]
            return f"Região prevista: {regiao}"

        def gerar_graficos_feriados(subsistema, mostrar_feriados, comparar_feriados, data_inicio, data_fim):
            df_sub = df_feriados[df_feriados['nom_subsistema'] == subsistema]

            if data_inicio:
                df_sub = df_sub[df_sub['din_instante'] >= pd.to_datetime("2020-"+data_inicio)]
                print(data_inicio)
            if data_fim:
                df_sub = df_sub[df_sub['din_instante'] <= pd.to_datetime("2020-"+data_fim)]

            if comparar_feriados:
                fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
                ax1, ax2 = axs[0], axs[1]
            else:
                fig, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
                ax2 = None

            ax1.plot(df_sub['din_instante'], df_sub['val_cargaenergiamwmed'], label='Carga', color='blue')
            ax1.set_title(f"Consumo Diário - {subsistema} (2020)")
            ax1.set_ylabel("MW médios")
            ax1.set_xlabel("Data")

            if mostrar_feriados:
                feriados = df_sub[df_sub['feriado']]
                ax1.scatter(feriados['din_instante'], feriados['val_cargaenergiamwmed'], color='red', label='Feriado', zorder=5)
            ax1.legend()
            ax1.grid(True)

            if comparar_feriados and ax2:
                feriado_medio = df_sub[df_sub['feriado']]['val_cargaenergiamwmed'].mean()
                normal_medio = df_sub[~df_sub['feriado']]['val_cargaenergiamwmed'].mean()

                categorias = ['Dias Normais', 'Feriados']
                valores = [normal_medio, feriado_medio]

                ax2.bar(categorias, valores, color=['green', 'red'])
                ax2.set_title("Média de Consumo: Feriado vs Dia Normal")
                ax2.set_ylabel("MW médios")
                ax2.grid(True, axis='y')

            return fig

        colunas_numericas = df.drop(columns=["Unnamed: 19", "hora", "mes", 'datetime', 'Data', 'Hora UTC']).columns.tolist()
        def gerar_grafico_meteorologico(variaveis, tipo_grafico, agregacao, titulo, grid):
            if not variaveis:
                return None
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

        # Criar abas "Previsão" e "Análise Avançada" primeiro para definir grupos
        with gr.Tab("Previsão de temperatura") as tab_previsao:
            with gr.Group(visible=False, elem_id="grupo_previsao") as grupo_previsao:
                gr.Markdown("## Previsão de Temperatura com Random Forest")
                inp1 = gr.Slider(0, 50, label="PRECIPITAÇÃO TOTAL, HORÁRIO (mm)")
                inp2 = gr.Slider(900, 1050, label="PRESSÃO ATMOSFERICA AO NIVEL DA ESTACAO")
                inp3 = gr.Slider(0, 100, label="UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)")
                inp4 = gr.Slider(0, 23, step=1, label="Hora do dia")
                inp5 = gr.Slider(1, 12, step=1, label="Mês")
                botao_prever = gr.Button("Prever Temperatura")
                saida_previsao = gr.Number(label="Temperatura Prevista (°C)")
                graf1 = gr.Plot(label="Dispersão")
                graf2 = gr.Plot(label="Importância")
                graf3 = gr.Plot(label="Erro")
                botao_prever.click(
                    prever_temperatura,
                    inputs=[inp1, inp2, inp3, inp4, inp5],
                    outputs=[saida_previsao, graf1, graf2, graf3]
                )
        with gr.Tab("Gráfico Metereológico"):
            with gr.Group(visible=False, elem_id="grupo_avancado") as grupo_avancado:
                variaveis = gr.CheckboxGroup(choices=colunas_numericas, label="Variáveis")
                tipo_grafico = gr.Radio(["Linha", "Barras", "Área"], label="Tipo de gráfico", value="Linha")
                agregacao = gr.Radio(["Hora", "Dia", "Mês"], label="Agregação", value="Hora")
                titulo = gr.Textbox(label="Título do gráfico (opcional)")
                grid = gr.Checkbox(label="Mostrar grade", value=True)
                botao = gr.Button("Executar Análise")
                grafico = gr.Plot(label="Gráfico Gerado")

                botao.click(
                    gerar_grafico_meteorologico,
                    inputs=[variaveis, tipo_grafico, agregacao, titulo, grid],
                    outputs=grafico
                )
        with gr.Tab("Feriados e Eventos"):
            with gr.Group(visible=False, elem_id="feriados") as grupo_feriados:
                entrada_subsistema = gr.Dropdown(label="Subsistema", choices=subsistemas, value=subsistemas[0])
                entrada_mostrar_feriados = gr.Checkbox(label="Mostrar Feriados no Gráfico", value=True)
                entrada_comparar = gr.Checkbox(label="Comparar Média Feriados vs Dias Normais", value=True)
                entrada_data_inicio = gr.Textbox(label="Data Início (opcional)")
                entrada_data_fim = gr.Textbox(label="Data Fim (opcional)")
                botao_gerar = gr.Button("Gerar Gráfico")
                saida_plot = gr.Plot()

                botao_gerar.click(
                    gerar_graficos_feriados,
                    inputs=[
                        entrada_subsistema,
                        entrada_mostrar_feriados,
                        entrada_comparar,
                        entrada_data_inicio,
                        entrada_data_fim
                    ],
                    outputs=saida_plot
                )
        with gr.Tab("Prever Região"):
            with gr.Group(visible=False, elem_id="prevRegiao") as grupo_previsao_regiao:
                carga_input = gr.Number(label="Carga de Energia (MW médios)")
                dia_semana_input = gr.Slider(1, 7, step=1, label="Dia da Semana (1=Dom, ..., 7=Sáb)")
                mes_input = gr.Slider(1, 12, step=1, label="Mês")
                botao_prever_regiao = gr.Button("Prever Região")
                saida_previsao_regiao = gr.Text(label="Previsão da Região")

                botao_prever_regiao.click(
                    prever_regiao,
                    inputs=[carga_input, dia_semana_input, mes_input],
                    outputs=saida_previsao_regiao
                )
        # Definir grupos de visibilidade para as abas    
        # Agora sim, defina a aba Login que usa esses grupos
        with gr.Tab("Sazonalidade"):
            with gr.Group(visible=False, elem_id="sazonalidade") as grupo_sazonalidade:

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

                gr.Markdown("## Decomposição Sazonal do Consumo no Nordeste (2020-2021)")
                escolha_grafico = gr.Radio(choices=["Decomposição Sazonal", "Padrão Sazonal Médio Mensal",  "Sazonalidade por Estação e Ano"], label="Escolha o gráfico:")
                output_plot = gr.Image(type="numpy")  # ou type="auto" se quiser deixar dinâmico

                escolha_grafico.change(fn=mostrar_grafico, inputs=escolha_grafico, outputs=output_plot)
                
        with gr.Tab("Crescimento Populacional"):
            with gr.Group(visible=False, elem_id="grupo_crescimento_populacional") as grupo_crescimento_populacional:
                gr.Markdown("## Análise do Crescimento Populacional em Recife (2020)")
                gr.Markdown("Esta seção analisa a população estimada mensalmente em Recife durante o ano de 2020, incluindo a taxa de crescimento populacional.")
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
                plot_type = gr.Dropdown(choices=["População", "Taxa de Crescimento"], label="Selecione o tipo de gráfico", value="População")
                table_output = gr.HTML()
                plot_output = gr.Plot()
                stats_output = gr.Markdown()

                plot_type.change(fn=generate_interface, inputs=plot_type, outputs=[table_output, plot_output, stats_output])
                app.load(fn=generate_interface, inputs=plot_type, outputs=[table_output, plot_output, stats_output])


        with gr.Tab("Login"):
            login_email = gr.Textbox(label="Email")
            login_senha = gr.Textbox(label="Senha", type="password")
            login_botao = gr.Button("Login")
            login_saida = gr.Text()
            mfa_email = gr.Textbox(label="Email", interactive=False, visible=False)
            mfa_token = gr.Textbox(label="Token MFA", visible=False)
            mfa_saida = gr.Text(visible=False)
            botao_mfa = gr.Button("Verificar MFA", visible=False)

            login_botao.click(
                autenticar,
                inputs=[login_email, login_senha],
                outputs=[login_saida, estado_login, email_login, mostrar_mfa]
            ).then(
                atualiza_mfa_visibilidade,
                inputs=[mostrar_mfa, email_login],
                outputs=[mfa_email, mfa_token, mfa_saida, botao_mfa]
            )

            botao_mfa.click(
                verificar_mfa,
                inputs=[mfa_email, mfa_token],
                outputs=[mfa_saida, estado_login]
            ).then(
                atualizar_visibilidade_grupos,
                inputs=[estado_login],
                outputs=[grupo_previsao, grupo_avancado, grupo_feriados, grupo_previsao_regiao, grupo_sazonalidade, grupo_crescimento_populacional]
            )

        with gr.Tab("Cadastro"):
            cadastro_email = gr.Textbox(label="Email")
            cadastro_senha = gr.Textbox(label="Senha", type="password")
            botao_cadastro = gr.Button("Cadastrar")
            saida_cadastro = gr.Text()
            qr_code = gr.Image()
            botao_cadastro.click(
                register,
                inputs=[cadastro_email, cadastro_senha],
                outputs=[saida_cadastro, qr_code]
            )

app.launch(server_name="0.0.0.0", server_port=8080)
