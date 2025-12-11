import pandas as pd

try:
    from .visualization import distribution_by_null_values, distribution_by_type_of_age_rating, distribution_by_type_of_event, distribution_by_type_of_session, histogram, relationship_between_ticket_price_and_quantity_sold
except ImportError:
    from visualization import distribution_by_null_values, distribution_by_type_of_age_rating, distribution_by_type_of_event, distribution_by_type_of_session, histogram, relationship_between_ticket_price_and_quantity_sold

def data_exploration(data_path):
    # Configurar pandas para melhor visualização
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    data = pd.read_csv(data_path, sep=';', skiprows=1)

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

    colums_to_see = ["Espaço", "Tipo de Evento", "Classificação Etária", "Tipo da Sessão"]
    print("VALORES ÚNICOS POR COLUNA")
    for col in colums_to_see:
        print(f"{col}: {set(data[col])}")
    print("\n")

    histogram(data, "data_exploration")
    distribution_by_type_of_event(data, "data_exploration")
    distribution_by_type_of_age_rating(data, "data_exploration")
    distribution_by_type_of_session(data, "data_exploration")
    distribution_by_null_values(data, "data_exploration")
    relationship_between_ticket_price_and_quantity_sold(data, "data_exploration")

if __name__ == "__main__":
    data_path = "data/bilheteria.csv"
    data_exploration(data_path)
