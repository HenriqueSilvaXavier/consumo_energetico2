import gradio as gr
import json
import bcrypt
import pyotp
import qrcode
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
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
    