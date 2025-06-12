import pandas as pd
import matplotlib.pyplot as plt
import holidays
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



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