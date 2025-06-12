import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from autenticar import atualiza_mfa_visibilidade, atualizar_visibilidade_grupos, load_users, save_users, create_qr_code, register, autenticar, verificar_mfa, carregar_e_processar_dados
from graficos_metereologicos import gerar_grafico_meteorologico, colunas_numericas
from previsao_temperatura import  treinar_modelo, prever_temperatura, gerar_graficos 
from populacao import load_data, generate_interface
from sazonalidade import load_prepare_sazonalidade, gerar_decomposicao, gerar_grafico_barra, gerar_grafico_estacoes, mostrar_grafico
from feriados import gerar_graficos_feriados, subsistemas
from regiao import prever_regiao

with gr.Blocks() as app:
    estado_login = gr.State(False)
    email_login = gr.State("")
    mostrar_mfa = gr.State(False)
    with gr.Tabs() as abas:
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
                gr.Markdown("## Decomposição Sazonal do Consumo no Nordeste (2020-2021)")
                escolha_grafico = gr.Radio(choices=["Decomposição Sazonal", "Padrão Sazonal Médio Mensal",  "Sazonalidade por Estação e Ano"], label="Escolha o gráfico:")
                output_plot = gr.Image(type="numpy")  # ou type="auto" se quiser deixar dinâmico

                escolha_grafico.change(fn=mostrar_grafico, inputs=escolha_grafico, outputs=output_plot)
                
        with gr.Tab("Crescimento Populacional"):
            with gr.Group(visible=False, elem_id="grupo_crescimento_populacional") as grupo_crescimento_populacional:
                gr.Markdown("## Análise do Crescimento Populacional em Recife (2020)")
                gr.Markdown("Esta seção analisa a população estimada mensalmente em Recife durante o ano de 2020, incluindo a taxa de crescimento populacional.")
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

app.launch(server_name="127.0.0.1", server_port=7860)