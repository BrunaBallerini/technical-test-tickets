import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import unicodedata
import re
from visualization import visualize_data

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

if __name__ == "__main__":

    data = pd.read_csv('data/bilheteria.csv', sep=';', skiprows=1)

    # Remover a última coluna se ela estiver completamente vazia
    if data.iloc[:, -1].isnull().all():
        data = data.iloc[:, :-1]

    # Remover todas as linhas que contêm algum valor nulo
    data = data.dropna()

    # Aplicar One-Hot Encoding nas colunas categóricas (nomes já normalizados)
    colunas_categoricas = ['Espaço', 'Evento', 'Tipo de Evento', 'Classificação Etária', 'Tipo da Sessão']

    # Separar as colunas numéricas das categóricas
    colunas_numericas = [col for col in data.columns if col not in colunas_categoricas]

    # Inicializar o OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Aplicar one-hot encoding nas colunas categóricas
    encoded_array = encoder.fit_transform(data[colunas_categoricas])

    # Obter os nomes das novas colunas
    feature_names = encoder.get_feature_names_out(colunas_categoricas)

    # Criar DataFrame com as colunas encodadas (converter matriz esparsa para densa)
    data_encoded = pd.DataFrame(encoded_array.toarray(), columns=feature_names, index=data.index)

    # Concatenar com as colunas numéricas
    final_data = pd.concat([data[colunas_numericas], data_encoded], axis=1)

    # Normalizar nomes das colunas
    final_data = normalize_column_names(final_data)

    # Salvar o dataset processado
    final_data.to_csv('data/bilheteria_processado.csv', index=False)

    visualize_data(final_data, "preprocessing")