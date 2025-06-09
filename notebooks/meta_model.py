import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error

class SequentialEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, base_estimator, n_estimators=50, sample_size=0.75, lr=0.1,
                 random_state=None, early_stopping=False, patience=5):
        """
        Meta-modelo de ensamble secuencial para regresión con early stopping opcional.

        Parameters:
        - base_estimator: objeto scikit-learn, modelo base (regresor), ya configurado
        - n_estimators: int, número máximo de modelos a ensamblar
        - sample_size: float, proporción de datos a usar en cada iteración (entre 0 y 1, sin reemplazo)
        - lr: float, tasa de aprendizaje
        - random_state: int o None, semilla para reproducibilidad
        - early_stopping: bool, si True se detiene el entrenamiento si no hay mejora en validación
        - patience: int, número de iteraciones sin mejora antes de detener (solo si early_stopping=True)
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.lr = lr
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience

    def fit(self, X, y):
        """
        Entrena secuencialmente un conjunto de modelos para minimizar el error cuadrático.

        En cada iteración:
        - Calcula el residuo (error) entre la predicción actual y y
        - Entrena un nuevo modelo sobre una muestra aleatoria sin reemplazo
        - Suma su predicción ponderada por la tasa de aprendizaje
        - (Opcional) Evalúa en los datos fuera de la muestra para aplicar early stopping
        """
        rng = check_random_state(self.random_state)
        self.models_ = []
        self.init_pred_ = np.mean(y)  # pred0 inicial
        pred_actual = np.full_like(y, self.init_pred_, dtype=float)
        residual = y - pred_actual

        best_val_mse = float('inf')
        no_improve_count = 0
        self.val_errors_ = []

        for i in range(self.n_estimators):
            # Muestreo aleatorio sin reemplazo
            idx = rng.choice(len(X), size=int(len(X) * self.sample_size), replace=False)
            X_train, y_train = X[idx], residual[idx]

            # Entrenamiento del modelo sobre el residuo
            model = clone(self.base_estimator)
            model.fit(X_train, y_train)

            # Predicción y actualización del ensamble
            pred_i = model.predict(X)
            pred_actual += self.lr * pred_i
            residual = y - pred_actual

            self.models_.append(model)

            # Evaluación en datos fuera de la muestra para early stopping
            if self.early_stopping and self.sample_size < 1.0:
                mask = np.ones(len(X), dtype=bool)
                mask[idx] = False  # datos no usados en esta iteración
                if np.any(mask):
                    X_val, y_val = X[mask], y[mask]
                    pred_val = self.predict(X_val)
                    val_mse = mean_squared_error(y_val, pred_val)
                    self.val_errors_.append(val_mse)

                    if val_mse < best_val_mse - 1e-6:
                        best_val_mse = val_mse
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if no_improve_count >= self.patience:
                        break

        return self

    def predict(self, X):
        """
        Calcula la predicción final del meta-modelo sobre un conjunto de datos.

        Predicción = pred0 + lr * (modelo_1(X) + modelo_2(X) + ... + modelo_n(X))
        """
        pred = np.full(X.shape[0], self.init_pred_, dtype=float)
        for model in self.models_:
            pred += self.lr * model.predict(X)
        return pred