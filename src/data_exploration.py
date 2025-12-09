import pandas as pd

from visualization import visualize_data

# Configurar pandas para melhor visualização
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

data = pd.read_csv('data/bilheteria.csv', sep=';', skiprows=1)

print("PRIMEIRAS LINHAS DO DATASET")
print(data.head())
print("\n")

print("INFORMAÇÕES DO DATASET")
print(data.info())
print("\n")

print("DIMENSÕES DO DATASET")
print(f"Linhas: {data.shape[0]} x Colunas: {data.shape[1]}")
print("\n")

print("COLUNAS DO DATASET")
for col in data.columns:
    print(f"{col}")
print("\n")

print("VALORES NULOS POR COLUNA")
for col, count in data.isnull().sum().items():
    print(f"{col}: {count}")
print("\n")

print("DESCRIÇÃO DO DATASET")
print(data.describe())
print("\n")

set_of_columns = set(data.columns)
colums_to_see = ["Espaço", "Tipo de Evento", "Classificação Etária", "Tipo da Sessão"]
print("VALORES ÚNICOS POR COLUNA")
for col in colums_to_see:
    print(f"{col}: {set(data[col])}")
print("\n")

visualize_data(data, "data_exploration")
