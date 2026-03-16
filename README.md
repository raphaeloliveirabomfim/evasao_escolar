# Previsão de Evasão Escolar no Ensino Médio Público Brasileiro

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-blue)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Visão Geral

Projeto de Machine Learning para **identificar precocemente alunos do Ensino Médio público com alto risco de evasão escolar**, permitindo que escolas e secretarias de educação atuem de forma proativa — antes que o abandono ocorra.

> A evasão escolar atinge cerca de **11-22% dos alunos do Ensino Médio** na rede pública brasileira (INEP, 2022), perpetuando ciclos de desigualdade social e econômica.

---

## Problema

A abordagem tradicional é **reativa**: o aluno abandona e a escola tenta reintegrar. Com dados históricos e ML, mudamos para uma abordagem **preditiva e proativa**:

> *"No início do ano letivo, quais alunos têm maior probabilidade de evadir?"*

---

## Estrutura do Projeto

```
evasao-escolar/
│
├── notebooks/
│   ├── 01_geracao_dados.ipynb     # Geração do dataset sintético baseado no INEP
│   ├── 02_eda.ipynb               # Análise Exploratória de Dados
│   ├── 03_modelagem.ipynb         # Feature engineering, pipelines e avaliação
│   └── 04_interpretacao.ipynb     # Interpretação e recomendações
│
├── data/
│   ├── evasao_escolar.csv         # Dataset gerado (15.000 alunos)
│   └── alunos_com_risco.csv       # Output: scores de risco por aluno
│
├── figures/                     # Gráficos gerados automaticamente
│
├── models/
│   ├── pipeline_xgb.pkl           # Modelo final (pipeline completo)
│   ├── threshold_xgb.pkl          # Threshold otimizado por F2-Score
│   └── resultados_modelos.csv     # Tabela comparativa de métricas
│
├── requirements.txt
└── README.md
```

---

## Dados

Os dados utilizados são **sintéticos**, gerados com distribuições baseadas em estatísticas reais:

| Fonte | Uso |
|-------|-----|
| [INEP — Censo Escolar 2022](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar) | Taxas de evasão, distorção idade-série |
| [SAEB 2021](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/saeb) | Proficiências médias em Português e Matemática |
| [IBGE — PNAD Contínua 2022](https://www.ibge.gov.br/estatisticas/sociais/trabalho/9173-pesquisa-nacional-por-amostra-de-domicilios-continua-trimestral.html) | Renda familiar, escolaridade |

**Por que dados sintéticos?** Os microdados do INEP têm vários GBs e exigem cadastro institucional. Os dados sintéticos preservam as relações estatísticas documentadas e garantem reprodutibilidade imediata do projeto.

---

## Metodologia

### Pipeline

```
Dados - Feature Engineering: Pré-processamento (StandardScaler + OHE)
      - Divisão Treino/Teste (stratify): SMOTE (apenas treino)
      - Treinamento (LR, RF, XGB): Tuning de Threshold (F2-Score)
      - Avaliação Final: Modelo Salvo
```

### Feature Engineering

| Feature | Descrição |
|---------|-----------|
| `media_notas` | Média de Português e Matemática |
| `baixo_desempenho` | Nota < 5 em qualquer disciplina |
| `risco_faltas` | Faltas anuais > 15 dias |
| `risco_distorcao` | Distorção ≥ 2 anos |
| `engajamento` | Proxy: média_notas / (faltas + 1) |

### Métricas e Escolha do Modelo

Para este problema, **Recall é a métrica prioritária**: um falso negativo (aluno em risco não identificado) tem custo humano e social muito maior que um falso positivo (atenção desnecessária a um aluno saudável).

Usei o **F2-Score** (que pondera o Recall 2× mais que a Precisão) para selecionar o threshold ótimo de decisão.

---

## Resultados

| Modelo | Threshold | Recall | Precisão | F2-Score | AUC-ROC |
|--------|-----------|--------|----------|----------|---------|
| Logistic Regression | 0.385 | 0.906 | 0.589 | 0.818 | 0.930 |
| Random Forest | 0.314 | 0.937 | 0.698 | 0.877 | 0.968 |
| **XGBoost**  | 0.382 | 0.953 | 0.6826 | 0.883 | 0.970 |

> *Execute o notebook `03_modelagem.ipynb` para ver os resultados completos com os valores reais.*

---

## Principais Fatores de Risco

1. **Distorção idade-série** — alunos com 2+ anos de atraso têm risco 4-5× maior
2. **Faltas anuais** — acima de 20 dias, o risco aumenta drasticamente
3. **Nota de Matemática** — correlação negativa forte (–0.41)
4. **Trabalhar** — especialmente no turno noturno
5. **Reprovações anteriores** — histórico de repetência

---

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/evasao-escolar.git
cd evasao-escolar

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute os notebooks em ordem
jupyter notebook notebooks/01.geracao_dados.ipynb
jupyter notebook notebooks/02.eda.ipynb
jupyter notebook notebooks/03.modelagem.ipynb
jupyter notebook notebooks/04.interpretacao.ipynb
```

---

## Próximos Passos

- [ ] Validar com microdados reais do Censo Escolar 2022/2023
- [ ] Dashboard Streamlit para gestores escolares
- [ ] API REST (FastAPI) para integração com sistemas de gestão escolar
- [ ] SHAP values para explicabilidade individual por aluno
- [ ] Expansão para segmentação municipal

---

## Autor

**Raphael Oliveira Bomfim**
- LinkedIn: [https://www.linkedin.com/in/raphael-oliveira-bomfim/](https://linkedin.com)
- GitHub: [https://github.com/raphaeloliveirabomfim](https://github.com)
- Email: rapha.sep@hotmail.com
  
Desenvolvido como projeto de portfólio de Ciência de Dados.

---

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

