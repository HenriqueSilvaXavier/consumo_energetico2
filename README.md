# ⚡ Otimizador de Consumo Energético Residencial

Sistema inteligente que utiliza **machine learning**, **dados climáticos e demográficos** para prever picos de consumo de energia residencial, com foco em segurança da informação e implantação em nuvem (Render).

---

## 🎯 Objetivo

Prever padrões de consumo energético com base em dados históricos e variáveis ambientais (clima, população, sazonalidade), oferecendo suporte à gestão eficiente da distribuição de energia. O sistema também garante segurança via autenticação multifatorial (MFA) e está disponível via cloud.

---

## 🔐 Segurança da Informação

- ✅ Autenticação multifatorial (MFA) via Google Authenticator (TOTP)
- ✅ Senhas seguras com hash (bcrypt)
- ✅ Armazenamento seguro e auditável de usuários (`users.json`)
- 🔄 Monitoramento contínuo (em desenvolvimento via logs)

---

## ☁️ Cloud Computing

- ✅ Deploy na nuvem via [Render](https://render.com/)

---

## 🌐 Acesso Online

A aplicação está disponível via Render:

🔗 **URL de produção:** [https://otimizador-energia.onrender.com](https://consumo-energetico2-6.onrender.com/)

---

## 📁 Estrutura do Projeto

| Arquivo                            | Descrição |
|-----------------------------------|-----------|
| `app.py`                          | Backend com rotas de autenticação e previsão |
| `users.json`                      | Banco de usuários com MFA |
| `CARGA_ENERGIA_2020.csv`          | Dados de carga elétrica |
| `CONSUMO_energia.csv`             | Dados gerais de consumo |
| `INMET_NE_PE_A301_2020.csv`       | Clima (INMET) - Recife |
| `populacao_recife_2020.csv`       | Crescimento populacional |
| `requirements.txt`                | Dependências Python |
| `Dockerfile`                      | Build da aplicação para container |
| `start.sh`                        | Script de inicialização |
| `Procfile`                        | Para compatibilidade com deploys |

---

## 🔍 Funcionalidades

- Previsão de consumo energético
- Cadastro com MFA via QR Code (Google Authenticator)
- Login seguro com senha + token MFA
- Visualização e integração com dados meteorológicos e populacionais

---

## 🧠 Features para o modelo

- Consumo histórico
- Temperatura e umidade
- Estações do ano e feriados
- Crescimento populacional
- Eventos pontuais

---

## 🔐 Fluxo de Login com MFA

2. **Login:** usuário entra com email e senha
3. **Verificação MFA:** digitação do token de 6 dígitos
4. **Acesso autorizado**

---

## ▶️ Execução Local

```bash
git clone https://github.com/seu-usuario/otimizador-energia.git
cd otimizador-energia

python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

pip install -r requirements.txt

python app.py

---
```

## 📊 Relatório de Gráficos

O relatório a seguir apresenta as visualizações gráficas geradas a partir dos dados utilizados no projeto:

📄 **Relatório de Gráficos:** [Clique aqui para visualizar](https://docs.google.com/document/d/1QVsBIPr93vxZMiE85ktjTHjnSghlBFm-ecSB3PuRYtw/edit?usp=sharing)




