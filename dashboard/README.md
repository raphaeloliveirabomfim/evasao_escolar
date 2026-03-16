# Dashboard — Evasão Escolar

Duas versões do dashboard estão disponíveis nesta pasta:

---

## 1 · Streamlit (`app.py`)

Dashboard interativo em Python, rodando localmente.

### Como executar

```bash
# 1. A partir da raiz do projeto, instale as dependências do dashboard
pip install -r dashboard/requirements_dashboard.txt

# 2. Execute — certifique-se de que os notebooks já foram rodados
#    (os arquivos data/ e models/ precisam existir)
streamlit run dashboard/app.py
```

O app abre automaticamente em `http://localhost:8501`

### Funcionalidades

| Aba                  | Conteúdo                                                  |
|----------------------|-----------------------------------------------------------|
|  Visão Geral       | KPIs, evasão por série/turno/distorção, scatter plot      |
|  Perfil de Risco   | Distribuição de risco, boxplots, heatmap                  |
|  Desempenho        | Métricas do modelo, feature importance, validação         |
|  Prever Aluno      | Formulário → score de risco individual com gauge          |

### Pré-requisitos

Execute os notebooks em ordem antes de rodar a dashboard:
1. `notebooks/01_geracao_dados.ipynb` → gera `data/evasao_escolar.csv`
2. `notebooks/03_modelagem.ipynb`     → gera `models/pipeline_rf.pkl`

---

## Estrutura de arquivos

```
dashboard/
├── app.py                      # App Streamlit completo
├── requirements_dashboard.txt  # Dependências do Streamlit
└── README.md                   # Este arquivo
```
