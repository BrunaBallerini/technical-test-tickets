import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_single_split(model, X_train, X_test, y_train, y_test, model_name):
    """
    Avalia modelo usando train_test_split
    """
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{model_name} - Single Split")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.1f}")
    print(f"RMSE: {rmse:.1f}")
    
    return model


def evaluate_cross_validation(model, X, y, model_name):
    """
    Avalia usando cross_validate (MUITO mais rápido que 3 cross_val_score)
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_rmse': 'neg_root_mean_squared_error'
    }
    
    scores = cross_validate(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
    
    mean_r2 = np.mean(scores['test_r2'])
    mean_mae = -np.mean(scores['test_neg_mae'])
    mean_rmse = -np.mean(scores['test_neg_rmse'])

    print(f"\n{model_name} - Cross Validation - 5 Folds")
    print(f"R² Médio:   {mean_r2:.3f}")
    print(f"MAE Médio:  {mean_mae:.1f}")
    print(f"RMSE Médio: {mean_rmse:.1f}")
    
    return mean_r2, mean_mae, mean_rmse


def print_feature_importance(model, feature_names, model_name):
    """
    Mostra as variáveis mais importantes para o modelo
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1] # Ordenar decrescente
        
        print(f"\nVariáveis importantes para {model_name}:")
        for i in range(min(5, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


def save_model(model, model_name, output_dir='models'):
    """
    Salva o modelo treinado e informações relacionadas
    
    Args:
        model: Modelo treinado
        model_name: Nome do modelo
        output_dir: Diretório onde salvar o modelo
    """
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar o modelo
    model_filename = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    joblib.dump(model, model_filename)
    print(f"\nModelo salvo em: {model_filename}")



def load_model(model_path):
    """
    Carrega um modelo salvo
    
    Args:
        model_path: Caminho para o arquivo do modelo
    
    Returns:
        model: Modelo carregado
    """
    model = joblib.load(model_path)
    print(f"Modelo carregado de: {model_path}")
    
    return model


def modeling(data_path, output_dir='models'):
    """
    Função que modela o dataset
    
    Args:
        data_path: Caminho para os dados processados
        save_best_model: Se True, salva o melhor modelo
        output_dir: Diretório onde salvar o modelo
    
    Returns:
        best_model: O melhor modelo treinado
        feature_names: Lista com os nomes das features
    """

    processed_data = pd.read_csv(data_path, sep=',')
    
    X = processed_data.drop(columns=['quantidade_de_ingressos_vendidos'])
    y = processed_data['quantidade_de_ingressos_vendidos']
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,                # Aumentar a profundidade
            learning_rate=0.15,         # Learning rate mais alto
            subsample=0.9,              # Aumentar subsample
            min_samples_split=10,       # Mais regularização
            min_samples_leaf=4,         # Mais regularização
            max_features='sqrt',        # Reduzir correlação entre árvores
            random_state=42
        )
    }
    
    results_cv = {}
    trained_models = {}
    
    for name, model in models.items():
        r2_cv, mae_cv, rmse_cv = evaluate_cross_validation(model, X, y, name)
        results_cv[name] = {'r2': r2_cv, 'mae': mae_cv}
    
        trained_model = evaluate_single_split(model, X_train, X_test, y_train, y_test, name)
        trained_models[name] = trained_model
        print_feature_importance(trained_model, feature_names, name)

    print("\nComparação dos modelos")

    results_sorted = sorted(results_cv.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for name, metrics in results_sorted:
        print(f"{name:<20} | R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.1f}")
       
    best_model_name = results_sorted[0][0]
    print(f"Melhor modelo: {best_model_name}")
    
    # Salvar o melhor modelo
    best_model = trained_models[best_model_name]

    save_model(best_model, best_model_name, output_dir)


if __name__ == "__main__":
    data_path = "data/bilheteria_processado.csv"
    
    modeling(data_path)
