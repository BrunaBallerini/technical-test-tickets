# Análise de Bilheteria FUNARTE

## Problema: Prever vendas de ingressos

## data_exploration.py

- **Dataset original:** 538 registros com 12 colunas
- **Valores nulos identificados:** Múltiplas colunas apresentavam valores ausentes
- **Primeira linha:** Continha metadados e foi removida (`skiprows=1`)
- **Última coluna:** Estava completamente vazia e foi removida
- **Registros zerados:** Diversas linhas continham valores zerados em campos críticos
- **Tipos de Evento:** Teatro, Dança, Música, Circo, Artes Integradas e Outras
- **Classificação Etária:** Livre, Adulto e Infantil
- **Tipos de Sessão:** Aberta e Fechada
- **Espaços:** Múltiplos espaços culturais da FUNARTE em RJ, SP e MG
- Variáveis categóricas que requerem aplicar tecnica OneHotEncoder
- Relação entre valor do ingresso e quantidade vendida apresenta dispersão
- Distribuição não uniforme entre tipos de eventos

### Visualizações Geradas
- **01_histogramas.png** - Distribuição de todas as variáveis numéricas
- **02_tipo_evento.png** - Distribuição por tipo de evento
- **03_classificacao_etaria.png** - Distribuição por classificação etária
- **04_tipo_sessao.png** - Distribuição por tipo de sessão
- **05_valores_nulos.png** - Quantidade de valores nulos por coluna
- **06_valor_vs_quantidade.png** - Relação entre valor do ingresso e quantidade vendida
- **07_heatmap_correlacao.png** - Matriz de correlação (Spearman)

## preprocessing.py

- Remoção da primeira linha (metadados)
- Remoção da última coluna vazia
- Remoção de todas as linhas com valores nulos
**Justificativa:** Valores nulos e zerados podem comprometer a qualidade dos modelos de machine learning. A remoção foi preferida à imputação devido à natureza dos dados de bilheteria, onde valores ausentes podem indicar sessões não realizadas ou dados não coletados.

- Normalização de Nomes de Colunas
**Justificativa:** Padronização evitando erros com caracteres especiais.

- One-Hot Encoding nos dados categóricos
**Justificativa:** Variáveis categóricas não podem ser diretamente utilizadas em modelos de machine learning. O One-Hot Encoding transforma cada categoria em uma coluna binária, preservando a informação sem criar relações ordinais artificiais.

- Dataset Processado salvo

## Próximos Passos (To-Do)

### Feature Engineering - Datas
- Extrair dia da semana das datas das sessões
- Criar features de sazonalidade (mês, semana do mês)
- Identificar finais de semana vs dias úteis
**Justificativa:** Padrões temporais podem revelar comportamentos de público (ex: maior público em finais de semana).

### Feature Engineering - Duração do Cartaz
- Calcular tempo de cartaz (data fim - data início)
- Analisar relação entre tempo de cartaz e vendas
- Identificar eventos de curta vs longa duração
**Justificativa:** A duração do evento pode influenciar estratégias de marketing e vendas.

### Tratamento de Data Leakage
- Avaliar remoção da coluna "Total de Vendas"
- Verificar se há outras variáveis derivadas
**Justificativa:** "Total de Vendas" é calculado como `Valor do Ingresso × Quantidade Vendida`, criando vazamento de informação que pode inflar artificialmente a performance do modelo.

### Modelagem
- Criar modelo baseline (ex: regressão linear)
- Implementar modelo mais complexo (ex: Random Forest, XGBoost)
- Validação cruzada temporal
- Análise de feature importance

### Análise Avançada
- Clustering de eventos similares
- Análise de séries temporais
- Identificação de eventos de "alto sucesso"
- Recomendações para otimização de bilheteria
