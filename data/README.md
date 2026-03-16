# Dados — Evasão Escolar

## Arquivos

| Arquivo | Descrição | Gerado por |
|---------|-----------|------------|
| `evasao_escolar.csv` | Dataset sintético com 15.000 alunos e 15 features | `01_geracao_dados.ipynb` |
| `alunos_com_risco.csv` | Dataset com scores de risco por aluno (output do modelo) | `04_interpretacao.ipynb` |

## Dicionário de Dados — `evasao_escolar.csv`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `idade` | int | Idade do aluno (14–24 anos) |
| `sexo` | str | M ou F |
| `raca_cor` | str | Autodeclaração racial (IBGE) |
| `serie` | str | 1ano, 2ano ou 3ano do EM |
| `turno` | str | matutino, vespertino ou noturno |
| `nota_portugues` | float | Nota de 0 a 10 |
| `nota_matematica` | float | Nota de 0 a 10 |
| `distorcao_idade_serie` | int | Anos de atraso escolar (0–5) |
| `faltas_anuais` | int | Número de dias de falta no ano |
| `repeticoes_anteriores` | int | Quantas vezes reprovou (0–3) |
| `trabalha` | int | 0 = não trabalha, 1 = trabalha |
| `renda_familiar` | int | Renda familiar em R$ (mensal) |
| `escolaridade_pai` | int | 0=fundamental, 1=médio, 2=superior, 3=pós |
| `escolaridade_mae` | int | Idem |
| `evasao` | int | **Target** — 0 = ficou, 1 = evadiu |

## Fontes de Referência

- [INEP — Censo Escolar](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar)
- [SAEB](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/saeb)
- [IBGE — PNAD Contínua](https://www.ibge.gov.br/estatisticas/sociais/trabalho/9173-pesquisa-nacional-por-amostra-de-domicilios-continua-trimestral.html)
