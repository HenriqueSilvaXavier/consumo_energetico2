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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
        "precipitacao": precipitacao,
        "pressao": pressao,
        "umidade": umidade,
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
            gr.update(visible=autenticado)
        )

    with gr.Tabs() as abas:
        print(df.columns)
        colunas_numericas = df.drop(columns=["Unnamed: 19", "hora", "mes", 'datetime', 'Data', 'Hora UTC']).columns.tolist()
        def gerar_grafico_meteorologico(variaveis, tipo_grafico, agregacao, titulo, grid):
            if not variaveis:
                return None
            if agregacao == "Hora":
                df_agg = df[["Data", "Hora UTC"] + variaveis].copy()
                df_agg["DataHora"] = pd.to_datetime(df_agg["Data"].dt.strftime("%Y-%m-%d") + " " + df_agg["Hora UTC"], errors='coerce')
                dados = df_agg.dropna(subset=["DataHora"] + variaveis)
                media = dados.groupby(dados["DataHora"].dt.hour)[variaveis].mean()
                eixo_x = media.index
                xlabel = "Hora do Dia (UTC)"
            elif agregacao == "Dia":
                dados = df[["Data"] + variaveis].dropna()
                media = dados.groupby("Data")[variaveis].mean()
                eixo_x = media.index
                xlabel = "Data"
            else:  # Mês
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
        with gr.Tab("Previsão") as tab_previsao:
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
        with gr.Tab("Análise Avançada"):
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
        # Agora sim, defina a aba Login que usa esses grupos
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
                outputs=[grupo_previsao, grupo_avancado]
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

app.launch()
