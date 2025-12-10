import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def histogram(data, folder_name):
    """
    Função que gera histogramas das variáveis numéricas
    
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None
    """

    sns.set_style("whitegrid")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
        data.hist(bins=50, figsize=(15, 8))
        plt.suptitle('Histogramas das Variáveis Numéricas', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(f'outputs/{folder_name}/01_histogramas.png')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar histogramas: {e}")


def distribution_by_type_of_event(data, folder_name):
    """
    Função que gera distribuição por tipo de evento
    
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None
    """

    sns.set_style("whitegrid")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
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


def distribution_by_type_of_age_rating(data, folder_name):
    """
    Função que gera distribuição por classificação etária
    
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None
    """

    sns.set_style("whitegrid")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
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


def distribution_by_type_of_session(data, folder_name):
    """
    Função que gera distribuição por tipo de sessão
    
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None
    """

    sns.set_style("whitegrid")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
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


def distribution_by_null_values(data, folder_name):
    """
    Função que gera distribuição por valores nulos
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None
    """

    sns.set_style("whitegrid")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
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


def relationship_between_ticket_price_and_quantity_sold(data, folder_name):
    """
    Função que gera relação entre valor do ingresso e quantidade vendida
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None 
    """

    sns.set_style("whitegrid")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
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


def correlation_heatmap(data, folder_name):
    """
    Função que gera heatmap de correlação
    Args:
        data: DataFrame com as variáveis numéricas
        folder_name: Nome da pasta para salvar os gráficos

    Returns:
        None
    """

    sns.set_style("white")
    os.makedirs(f'outputs/{folder_name}', exist_ok=True)

    try:
        corr_matrix = data.corr(method='spearman', numeric_only=True)
        fig, ax = plt.subplots(figsize=(24, 20))
        sns.heatmap(
            corr_matrix, 
            cmap='RdYlGn',  # Colormap com melhor contraste (vermelho-amarelo-verde)
            center=0,  # Centralizar o colormap no zero
            annot=False,  # Remover anotações numéricas
            fmt='.2f',
            square=True,  # Células quadradas
            linewidths=0.5,  # Linhas entre células
            linecolor='white',
            cbar_kws={
                'shrink': 0.8,
                'label': 'Correlação de Spearman',
                'orientation': 'vertical'
            },
            ax=ax
        )
        
        # Configurar título
        plt.title('Matriz de Correlação', fontsize=20, pad=20, fontweight='bold')
        
        # Configurar labels do eixo X
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha='right',
            fontsize=8,
            rotation_mode='anchor'
        )
        
        # Configurar labels do eixo Y
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=8
        )
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar com alta resolução
        plt.savefig(
            f'outputs/{folder_name}/07_heatmap_correlacao.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
    except Exception as e:
        print(f"Erro ao gerar heatmap de correlação: {e}")

