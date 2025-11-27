# Dataset: Bilheteria FUNARTE

## Descrição
Base de dados de bilheteria dos espaços culturais da FUNARTE (Fundação Nacional de Artes) contendo informações agregadas sobre eventos culturais e suas respectivas sessões, resultantes de propostas selecionadas via editais de ocupação.

## Período
Julho a Outubro de 2025

## Localização
Espaços da FUNARTE localizados no Rio de Janeiro, São Paulo e Minas Gerais.

## Estrutura dos Dados

### Arquivo: `bilheteria.csv`
- **Registros:** 538 sessões
- **Colunas:** 12 variáveis
- **Separador:** `;` (ponto e vírgula)
- **Encoding:** UTF-8

### Variáveis

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Espaço` | Categórica | Nome do espaço cultural |
| `Evento` | Categórica | Nome do evento |
| `Tipo de Evento` | Categórica | Teatro, Dança, Música, Circo, Artes Integradas, Outras |
| `Classificação Etária` | Categórica | Livre, Adulto, Infantil |
| `Período do Cartaz - Data Início` | Data/Hora | Início do período de exibição |
| `Período do Cartaz - Data Fim` | Data/Hora | Fim do período de exibição |
| `Tipo da Sessão` | Categórica | Fechada ou Aberta |
| `Data da Sessão` | Data/Hora | Data e horário específico da sessão |
| `Valor do Ingresso` | Numérica | Preço em R$ |
| `Quantidade de ingressos vendidos` | Numérica | Número de ingressos |
| `Total de Vendas` | Numérica | Receita total em R$ |

