import pandas as pd
import plotly.express as px
from feriados import df_feriados, rf, X, y, X_train, X_test, y_train, y_test, feature_cols, le

def prever_regiao(carga, dia_semana, mes):
    input_data = pd.DataFrame([[carga, dia_semana, mes]], columns=feature_cols)
    pred_label = rf.predict(input_data)[0]
    regiao = le.inverse_transform([pred_label])[0]
    return f"Região prevista: {regiao}"