# Previsão de Vendas de Ingressos - FUNARTE

Sistema de machine learning para prever a quantidade de ingressos vendidos em eventos culturais da FUNARTE.

## Visão Geral do Dataset

- **Registros:** 538 eventos
- **Colunas:** 12 variáveis (numéricas e categóricas)
- **Tipos de Evento:** Teatro, Dança, Música, Circo, Artes Integradas e Outras
- **Classificação Etária:** Livre, Adulto e Infantil
- **Tipos de Sessão:** Aberta e Fechada
- **Espaços:** Múltiplos espaços culturais da FUNARTE em RJ, SP e MG
- **Insights:**Variáveis categóricas que requerem aplicar tecnica OneHotEncoder

## Exploração de Dados (`data_exploration.py`)

Análise inicial identificou:
- Valores nulos em múltiplas colunas
- Primeira linha com metadados (removida)
- Última coluna vazia (removida)
- Necessidade de encoding para variáveis categóricas

**Visualizações geradas:** Histogramas, distribuições por categoria, matriz de correlação e análise de valores nulos.

## Pré-processamento (`preprocessing.py`)

### Limpeza de Dados
- **Valores de ingresso nulos:** Preenchidos com 0 (assumindo eventos gratuitos)
-- **Justificativa:** Anteriormente foi retirados todas as linhas com dados nulos, mas a quantidade de dados foi muito reduzida e não foi considerado os ingressos com valores nulos sendo.

- **Quantidade vendida nula:** Linhas removidas (variável target não pode ser nula)
-- **Justificativa:** Valores nulos e zerados podem comprometer a qualidade dos modelos de machine learning e a remoção foi escolhida para não tendencionar o modelo.

- **Coluna "Total de Vendas":** Removida para evitar data leakage (calculada como valor × quantidade)
-- **Justificativa:** Como a coluna total de vendas é calculado com o valor do ingresso × quantidade vendida, criando vazamento de informação que pode inflar 
artificialmente a performance do modelo.

### Engenharia de Features
- **One-Hot Encoding:** Aplicado em variáveis categóricas (exceto coluna "Evento" devido à alta cardinalidade)
-- **Justificativa:** Variáveis categóricas não podem ser diretamente utilizadas em modelos de machine learning. O One-Hot Encoding transforma cada categoria em uma coluna binária, preservando a informação sem criar relações ordinais artificiais. Primeiramente a coluna evento também passou pelo processo, mas aumento muito os dados e não trouxe informações pertinentes para o modelo.

- **Features temporais:** Extração de dia da semana, horário e tempo em cartaz
-- **Justificativa:** Padrões temporais podem revelar comportamentos de público, além de duração do evento pode trazer informações de sucesso do evento.

- **Normalização:** Padronização de nomes de colunas

**Resultado:** Dataset limpo salvo em `data/bilheteria_processado.csv`

## Modelagem (`modeling.py`)

### Modelos Testados
1. **Regressão Linear** (baseline)
2. **Random Forest**
3. **Gradient Boosting (XGBoost)**

### Técnicas Aplicadas
- Validação cruzada para avaliação robusta
- Análise de importância de features
- Seleção e salvamento do melhor modelo

**Modelos salvos em:** `models/`

## Como Usar

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 2. Pipeline Completo (Treinar Novo Modelo)

Execute todo o pipeline desde a exploração até o treinamento:

```bash
python main.py
```

**O script irá:**
1. Realizar exploração dos dados
2. Processar e limpar os dados
3. Treinar múltiplos modelos e salvar o melhor

### 3. Usar Modelo Já Treinado

Para fazer previsões com modelo treinado:

```bash
python exemplo_uso_modelo.py
```

Ou importe em seu código:

```python
import pandas as pd
from src.modeling import load_model

# Carregar modelo
model = load_model(
    'models/gradient_boosting_model.pkl',
)

# Carregar dados
data = pd.read_csv('data/bilheteria_processado.csv')
X_new = data.drop(columns=['quantidade_de_ingressos_vendidos'])

# Fazer previsões
predictions = model.predict(X_new)
print(f"Previsão de vendas: {predictions[0]:.0f} ingressos")
```

## Estrutura de Arquivos

```
├── data/
│   ├── bilheteria.csv                    # Dataset original
│   └── bilheteria_processado.csv         # Dados processados
├── instructions/
│   ├── evaluation_criteria.md
│   └── test_instructions.md
├── models/
│   └── gradient_boosting_model.pkl       # Melhor modelo
├── outputs/                              # Gráficos gerados em cada etapa
│   ├── data_exploration/
│   └── preprocessing/
├── src/
│   ├── data_exploration.py               # Análise exploratória
│   ├── preprocessing.py                  # Limpeza e transformação
│   ├── modeling.py                       # Treinamento e avaliação
│   └── visualization.py                  # Gráficos
└── main.py                               # Pipeline completo
```

## Notas

- Os dados de entrada devem seguir o mesmo formato e pré-processamento dos dados de treinamento
