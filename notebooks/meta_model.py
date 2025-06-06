import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score

class SequentialEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, n_estimators=50, sample_size=0.75, lr=0.1, random_state=None):
        """
        Meta-modelo de ensamble secuencial para regresión.

        Parameters:
        - base_estimator: objeto scikit-learn, modelo base (regresor), ya configurado
        - n_estimators: int, número de modelos a ensamblar
        - sample_size: float, proporción de datos a usar en cada iteración (entre 0 y 1)
        - lr: float, tasa de aprendizaje
        - random_state: int o None, semilla para reproducibilidad
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        self.models_ = []
        self.init_pred_ = np.mean(y)
        residual = y - self.init_pred_
        pred_actual = np.full_like(y, self.init_pred_, dtype=float)

        for _ in range(self.n_estimators):
            model = clone(self.base_estimator)
            idx = rng.choice(len(X), size=int(len(X) * self.sample_size), replace=False)
            X_sample, residual_sample = X[idx], residual[idx]
            model.fit(X_sample, residual_sample)
            pred_i = model.predict(X)
            pred_actual += self.lr * pred_i
            residual = y - pred_actual
            self.models_.append(model)

        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred_, dtype=float)
        for model in self.models_:
            pred += self.lr * model.predict(X)
        return pred
    
def explorar_hiperparametros(estimator_class, param_grid, X, y, cv):
    """
    Explora combinaciones de hiperparámetros para un meta-modelo de ensamble.

    Parameters:
    - estimator_class: clase del estimador base (ej. DecisionTreeRegressor)
    - param_grid: dict con listas de hiperparámetros a explorar (n_estimators, lr, sample_size, max_depth, etc.)
    - X, y: datos de entrenamiento
    - cv: generador de validación cruzada (KFold)

    Returns:
    - DataFrame con combinaciones y R² medio ordenado descendente
    """

    resultados = []

    n_estimators_list = param_grid.get("n_estimators", [50])
    lr_list = param_grid.get("lr", [0.1])
    sample_size_list = param_grid.get("sample_size", [0.8])
    max_depth_list = param_grid.get("max_depth", [None])

    for n_estimators in n_estimators_list:
        for lr in lr_list:
            for sample_size in sample_size_list:
                for max_depth in max_depth_list:
                    base_estimator = estimator_class()
                    if max_depth is not None:
                        base_estimator.set_params(max_depth=max_depth)

                    model = SequentialEnsembleRegressor(
                        base_estimator=base_estimator,
                        n_estimators=n_estimators,
                        lr=lr,
                        sample_size=sample_size,
                        random_state=42,
                    )

                    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                    resultados.append({
                        "n_estimators": n_estimators,
                        "lr": lr,
                        "sample_size": sample_size,
                        "max_depth": max_depth,
                        "r2_mean": np.round(r2_scores.mean(), 4)
                    })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados.sort_values(by="r2_mean", ascending=False)

def explorar_hiperparametros_knn(param_grid, X, y, cv):
    """
    Explora combinaciones de hiperparámetros para un meta-modelo de ensamble con KNeighborsRegressor.

    Parameters:
    - param_grid: dict con listas de hiperparámetros a explorar (n_estimators, lr, sample_size, n_neighbors, metric)
    - X, y: datos de entrenamiento
    - cv: generador de validación cruzada (KFold)

    Returns:
    - DataFrame con combinaciones y R² medio ordenado descendente
    """

    resultados = []

    n_estimators_list = param_grid.get("n_estimators", [50])
    lr_list = param_grid.get("lr", [0.1])
    sample_size_list = param_grid.get("sample_size", [0.8])
    n_neighbors_list = param_grid.get("n_neighbors", [5])
    metric_list = param_grid.get("metric", ['euclidean'])

    for n_estimators in n_estimators_list:
        for lr in lr_list:
            for sample_size in sample_size_list:
                for n_neighbors in n_neighbors_list:
                    for metric in metric_list:
                        base_estimator = KNeighborsRegressor(
                            n_neighbors=n_neighbors,
                            metric=metric
                        )

                        model = SequentialEnsembleRegressor(
                            base_estimator=base_estimator,
                            n_estimators=n_estimators,
                            lr=lr,
                            sample_size=sample_size,
                            random_state=42,
                        )

                        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                        resultados.append({
                            "n_estimators": n_estimators,
                            "lr": lr,
                            "sample_size": sample_size,
                            "n_neighbors": n_neighbors,
                            "metric": metric,
                            "r2_mean": np.round(r2_scores.mean(), 4)
                        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados.sort_values(by="r2_mean", ascending=False)
