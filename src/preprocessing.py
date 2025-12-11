import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import unicodedata
import re

try:
    from .visualization import correlation_heatmap
except ImportError:
    from visualization import correlation_heatmap

def normalize_column_names(df):
    """
    Normaliza os nomes das colunas:
    - Remove acentuação
    - Converte para minúsculas
    - Substitui espaços por underscore
    - Remove caracteres especiais
    """
    normalized_columns = []
    for col in df.columns:
        # Remove acentuação
        col_normalized = unicodedata.normalize('NFKD', col)
        col_normalized = col_normalized.encode('ascii', errors='ignore').decode('utf-8')
        
        # Converte para minúsculas
        col_normalized = col_normalized.lower()
        
        # Substitui espaços e caracteres especiais por underscore
        col_normalized = re.sub(r'[^\w\s]', '', col_normalized)
        col_normalized = re.sub(r'\s+', '_', col_normalized)
        
        normalized_columns.append(col_normalized)
    
    df.columns = normalized_columns
    return df


def extract_day_of_week(data):
    """
    Função que extrai o dia da semana da Data da Sessão
    Primeiro, extrai apenas a data (sem o horário) e converter para datetime
    Depois, extrai o dia da semana no formato: 0=Segunda, 1=Terça, ..., 6=Domingo
    Depois remove as colunas temporárias e retorna o dataframe
    Mantém a coluna 'Data da Sessão'
    Args:
        data: DataFrame com a coluna 'Data da Sessão'

    Returns:
        DataFrame com a coluna 'Dia da Semana'
    """
    data['data_sessao_temp'] = data['Data da Sessão'].str.split(' - ').str[0]
    data['data_sessao_dt'] = pd.to_datetime(data['data_sessao_temp'], format='%d/%m/%Y')
    data['Dia da Semana'] = data['data_sessao_dt'].dt.dayofweek
    data = data.drop(columns=['data_sessao_temp', 'data_sessao_dt'])

    return data

def get_hour_of_session(data):
    """
    Função que extrai a hora da sessão
    Extrai apenas a hora (0-23) da coluna 'Data da Sessão'
    
    Args:
        data: DataFrame com a coluna 'Data da Sessão'

    Returns:
        DataFrame com a coluna 'Hora'
    """
    temp_dates = pd.to_datetime(data['Data da Sessão'], format='%d/%m/%Y - %H:%M', errors='coerce')
    data['Hora'] = temp_dates.dt.hour

    return data


def extract_day_of_month(data):
    """
    Função que extrai o dia do mês da Data da Sessão
    Extrai apenas o dia (1-31) da coluna 'Data da Sessão'
    
    Args:
        data: DataFrame com a coluna 'Data da Sessão'

    Returns:
        DataFrame com a coluna 'Dia do Mês'
    """
    data['data_sessao_temp'] = data['Data da Sessão'].str.split(' - ').str[0]
    data['data_sessao_dt'] = pd.to_datetime(data['data_sessao_temp'], format='%d/%m/%Y')
    data['Dia do Mês'] = data['data_sessao_dt'].dt.day
    data = data.drop(columns=['data_sessao_temp', 'data_sessao_dt'])
    
    return data


def calculate_days_in_theaters(data):
    """
    Função que calcula os dias em cartaz
    Primeiro, extrai apenas as datas (formato DD/MM/YYYY) das colunas de período
    Depois converte para datetime
    Depois calcula a diferença em dias entre a data de início e a data de fim
    Depois remove as colunas temporárias e retorna o dataframe

    Args:
        data: DataFrame com as colunas 'Período do Cartaz - Data Início' e 'Período do Cartaz - Data Fim'

    Returns:
        DataFrame com a coluna 'Dias em Cartaz'
    """
    data['Período do Cartaz - Data Início'] = data['Período do Cartaz - Data Início'].str.split(' - ').str[0]
    data['Período do Cartaz - Data Fim'] = data['Período do Cartaz - Data Fim'].str.split(' - ').str[0]
    data['data_inicio'] = pd.to_datetime(data['Período do Cartaz - Data Início'], format='%d/%m/%Y')
    data['data_fim'] = pd.to_datetime(data['Período do Cartaz - Data Fim'], format='%d/%m/%Y')
    data['Dias em Cartaz'] = (data['data_fim'] - data['data_inicio']).dt.days
    data = data.drop(columns=['Período do Cartaz - Data Início', 'Período do Cartaz - Data Fim', 'data_inicio', 'data_fim'])

    return data

    
def one_hot_encoding(data, categorical_columns, numerical_columns):
    """
    Função que aplica One-Hot Encoding nas colunas categóricas
    Gera os nomes das novas colunas e concatena com as colunas numéricas

    Args:
        data: DataFrame com as colunas categóricas
        colunas_categoricas: Lista com as colunas categóricas
        colunas_numericas: Lista com as colunas numéricas

    Returns:
        DataFrame com as colunas categóricas codificadas
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_array = encoder.fit_transform(data[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    data_encoded = pd.DataFrame(encoded_array.toarray(), columns=feature_names, index=data.index)
    data = pd.concat([data[numerical_columns], data_encoded], axis=1)

    return data


def preprocessing(data_path):
    """
    Função que processa o dataset
    Primeiro, remove a coluna Total de Vendas para evitar overfitting
    Depois remove a última coluna se ela estiver completamente vazia
    Depois preenche o valor do ingresso vazio com 0 assumindo como gratuito
    Depois remove todas as linhas que contêm algum valor nulo na coluna 'Quantidade de ingressos vendidos'
    Depois extrai a hora da sessão
    Depois extrai o dia da semana
    Depois extrai o dia do mês
    Salva o dataset processado

    Args:
        data_path: Caminho do dataset

    Returns:
        None
    """
    # Configurar pandas para melhor visualização
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    try:
        data = pd.read_csv(data_path, sep=';', skiprows=1)

        if 'Total de Vendas' in data.columns:
            data = data.drop(columns=['Total de Vendas'])

        if data.iloc[:, -1].isnull().all():
            data = data.iloc[:, :-1]

        data['Valor do Ingresso'] = data['Valor do Ingresso'].fillna(0)

        data = data.dropna(subset=['Quantidade de ingressos vendidos'])

        data = get_hour_of_session(data)

        data = extract_day_of_week(data)
        
        data = extract_day_of_month(data)
        
        data = calculate_days_in_theaters(data)

        # Garante que função extract_day_of_week já foi aplicada e remove a coluna 'Data da Sessão'
        if 'Dia da Semana' in data.columns:
            data = data.drop(columns=['Data da Sessão'])

        # Separa as colunas categóricas das numéricas para aplicar One-Hot Encoding
        colunas_categoricas = ['Espaço', 'Tipo de Evento', 'Classificação Etária', 'Tipo da Sessão']
        colunas_numericas = [col for col in data.columns if col not in colunas_categoricas and col != 'Evento']

        data = one_hot_encoding(data, colunas_categoricas, colunas_numericas)

        # Normaliza nomes das colunas
        data = normalize_column_names(data)

        print("\nDataset processado com sucesso!")

        correlation_heatmap(data, "preprocessing")
        print("\nHeatmap de correlação gerado com sucesso!")
    except Exception as e:
        print(f"Erro ao processar o dataset: {e}")
        return None

    try:
        data.to_csv('data/bilheteria_processado.csv', index=False)
        print("\nDataset processado com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar o dataset processado: {e}")
        return None


if __name__ == "__main__":
    data_path = "data/bilheteria.csv"
    preprocessing(data_path)