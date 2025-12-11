# Análise de Bilheteria FUNARTE

## Problema 1: Prever vendas de ingressos

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
- **Insights:**Variáveis categóricas que requerem aplicar tecnica OneHotEncoder

### visualization.py
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
- Preencher o valor do ingresso vazio com 0 assumindo como gratuito
**Justificativa:** Anteriormente foi retirados todas as linhas com dados nulos, mas a quantidade de dados foi muito reduzida e não foi considerado os ingressos com valores nulos sendo.
- Remoção de todas as linhas com valores nulos da coluna quantidade de ingressos vendidos
**Justificativa:** Valores nulos e zerados podem comprometer a qualidade dos modelos de machine learning e a remoção foi escolhida para não tendencionar o modelo.
- Normalização de Nomes de Colunas
**Justificativa:** Padronização evitando erros com caracteres especiais.
- One-Hot Encoding nos dados categóricos menos da coluna Evento
**Justificativa:** Variáveis categóricas não podem ser diretamente utilizadas em modelos de machine learning. O One-Hot Encoding transforma cada categoria em uma coluna binária, preservando a informação sem criar relações ordinais artificiais. Primeiramente a coluna evento também passou pelo processo, mas aumento muito os dados e não trouxe informações pertinentes para o modelo.
- Remoção da coluna "Total de Vendas"
**Justificativa:** Como a coluan total de vendas é calculado comocom valor do ingresso × quantidade vendida, criando vazamento de informação que pode inflar artificialmente a performance do modelo.
- Extrair dia da semana das datas das sessões, deixando horário e dia do mês
- Cálculo do tempo que os eventos estão em cartaz
**Justificativa:** Padrões temporais podem revelar comportamentos de público, além de duração do evento pode trazer informações de sucesso do evento.
- Dataset Processado salvo

## modeling.py
- Modelo baseline de regressão linear
- Teste com modelos mais complexos: Random Forest e XGBoost
- Implantação da tecnica de validação cruzada
- Análise das features mais importantes por modelo
