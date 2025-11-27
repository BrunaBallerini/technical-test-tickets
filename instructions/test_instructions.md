# Instruções do Teste Técnico

## Contexto do Problema

Você está analisando uma base de dados da FUNARTE (Fundação Nacional de Artes). Em busca de otimizar a gestão de eventos culturais utilizando dados históricos de bilheteria de seus espaços culturais no Rio de Janeiro, São Paulo e Minas Gerais.

## Base de Dados

**Arquivo:** `data/bilheteria.csv`

**Descrição:** Dados agregados de eventos culturais e sessões de bilheteria de julho a outubro de 2025, incluindo:
- Informações do evento (nome, tipo, classificação etária)
- Dados do local e período
- Informações da sessão (data, hora, tipo)
- Dados financeiros (valor do ingresso, vendas, receita)

## Sugestões de Abordagem
A abordagem do problema tem um escopo aberto, sinta-se a vontade para utilizar as abordagens sugeridas ou qualquer outra abordagem que você julgar se encaixar bem no problema.

### Possíveis Problemas de Machine Learning
- **Regressão:** Prever número de ingressos vendidos
- **Classificação:** Identificar eventos de "alto sucesso"
- **Clustering:** Segmentar perfis de eventos
- **Séries Temporais:** Analisar sazonalidade

## Entregáveis Esperados

### 1. Análise Exploratória (EDA)
- **Estatísticas descritivas** das variáveis principais
- **Visualizações** que revelam padrões de negócio
- **insights** descobertas com base nos dados

### 2. Pré-processamento & Feature Engineering
- **Estratégia documentada** para dados faltantes
- **Criação de features** relevantes (temporais, categóricas, etc.)
- **Justificação** das transformações aplicadas

### 3. Modelagem
- **Modelo baseline** como referência (ex: regressão linear)
- **Modelo principal** com tuning apropriado
- **Validação robusta** (ex: cross-validation, split temporal e etc...)
- **Métricas de avaliação** claras e justificadas

### 4. Análise de Resultados & Recomendações
- **Interpretação** do modelo e features importantes
- **Limitações** limitações encontradas caso houver e possiveis próximos passos

## Formato de Entrega

### Opção 1: Notebook Jupyter
- Utilize o arquivo `notebook/challenge.ipynb` fornecido
- Documente seu processo de pensamento diretamente no notebook
- Inclua células markdown explicativas entre o código
- Apresente resultados e conclusões de forma clara

### Opção 2: Scripts Python
- Crie arquivos `.py` organizados por funcionalidade
- **Obrigatório:** Utilize o `README.md` para:
  - Explicar sua abordagem e metodologia
  - Documentar o processo de pensamento
  - Apresentar principais descobertas e resultados
  - Incluir instruções de execução

### Documentação Obrigatória
- **README.md:** Sempre deve conter instruções de uso e reprodução
- **requirements.txt:** Liste todas as dependências necessárias
- **Comentários no código:** Explique decisões técnicas importantes

### Estrutura Recomendada

projeto/
├── data/
│ └── bilheteria.csv
├── notebook/
│ └── challenge.ipynb (se usar notebook)
├── src/ (se usar scripts)
│ ├── eda.py
│ ├── preprocessing.py
│ └── modeling.py
├── README.md (obrigatório)
└── requirements.txt (obrigatório)

## Orientações Gerais

### Suposições e Decisões
- Você pode fazer suposições razoáveis sobre o negócio
- Documente todas as suas decisões e justificativas
- Foque na qualidade sobre quantidade
- Seja criativo mas mantenha o foco nos objetivos

### Qualidade do Código
- Utilize boas práticas de programação
- Comente o código adequadamente
- Organize o projeto de forma lógica
- Garanta a reprodutibilidade dos resultados

### Comunicação
- Use visualizações claras e informativas
- Explique metodologias e limitações
- Apresente recomendações acionáveis
