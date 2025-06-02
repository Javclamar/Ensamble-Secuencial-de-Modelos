import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state

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