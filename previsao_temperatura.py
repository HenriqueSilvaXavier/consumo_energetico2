import matplotlib.pyplot as plt
import pandas as pd
from autenticar import carregar_e_processar_dados
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import holidays
from sklearn.preprocessing import LabelEncoder

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