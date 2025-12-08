import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_data(data, folder_name):

    # Configurar estilo dos gráficos
    sns.set_style("whitegrid")

    # Criar pasta para salvar os gráficos
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
        # Histograma de todas as variáveis numéricas
        data.hist(bins=50, figsize=(15, 8))
        plt.suptitle('Histogramas das Variáveis Numéricas', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/01_histogramas.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar histogramas: {e}")

    try:
        # Distribuição por Tipo de Evento
        plt.figure(figsize=(10, 6))
        tipo_evento_counts = data['Tipo de Evento'].value_counts()
        plt.bar(range(len(tipo_evento_counts)), tipo_evento_counts.values, color='steelblue')
        plt.xticks(range(len(tipo_evento_counts)), tipo_evento_counts.index)
        plt.ylabel('Quantidade')
        plt.title('Distribuição por Tipo de Evento')
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/02_tipo_evento.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar distribuição por tipo de evento: {e}")

    try:
        # Distribuição por Classificação Etária
        plt.figure(figsize=(10, 6))
        classificacao_counts = data['Classificação Etária'].value_counts()
        plt.bar(range(len(classificacao_counts)), classificacao_counts.values, color='coral')
        plt.xticks(range(len(classificacao_counts)), classificacao_counts.index)
        plt.ylabel('Quantidade')
        plt.title('Distribuição por Classificação Etária')
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/03_classificacao_etaria.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar distribuição por classificação etária: {e}")

    try:
        # Distribuição por Tipo de Sessão
        plt.figure(figsize=(10, 6))
        sessao_counts = data['Tipo da Sessão'].value_counts()
        plt.bar(range(len(sessao_counts)), sessao_counts.values, color='mediumseagreen')
        plt.xticks(range(len(sessao_counts)), sessao_counts.index)
        plt.ylabel('Quantidade')
        plt.title('Distribuição por Tipo de Sessão')
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/04_tipo_sessao.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar distribuição por tipo de sessão: {e}")

    try:
        # Distribuição por valores nulos
        plt.figure(figsize=(10, 6))
        null_counts = data.isnull().sum()
        plt.bar(range(len(null_counts)), null_counts.values, color='red')   
        plt.xticks(range(len(null_counts)), null_counts.index, rotation=45, ha='right')
        plt.ylabel('Quantidade')
        plt.title('Distribuição por Valores Nulos')
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/05_valores_nulos.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar distribuição por valores nulos: {e}")

    try:
        # Relação entre Valor do Ingresso e Quantidade Vendida
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Valor do Ingresso'], data['Quantidade de ingressos vendidos'], alpha=0.5)
        plt.title("Relação entre Valor do Ingresso e Quantidade Vendida")
        plt.xlabel("Valor do Ingresso (R$)")
        plt.ylabel("Quantidade de Ingressos Vendidos")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'outputs/{folder_name}/06_valor_vs_quantidade.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar relação entre valor do ingresso e quantidade vendida: {e}")

    try:
        # Heatmap de Correlação (Spearman)
        plt.figure(figsize=(20, 10))
        sns.heatmap(data.corr(method='spearman', numeric_only=True), cmap='Greens', annot=True)
        plt.title('Matriz de Correlação', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/07_heatmap_correlacao.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar heatmap de correlação: {e}")

