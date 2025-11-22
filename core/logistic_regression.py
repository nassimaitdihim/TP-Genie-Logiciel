import numpy as np


class LogisticRegression:
    """Implémentation d'une régression logistique avec descente de gradient"""

    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        """Fonction sigmoid"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """Entraîne le modèle sur les données X et y"""
        n_samples, n_features = X.shape

        # Initialisation des paramètres
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Descente de gradient
        for i in range(self.n_iterations):
            # Prédiction
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Calcul des gradients avec régularisation L2
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (
                self.regularization * self.weights
            )
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calcul de la loss (log loss)
            if i % 100 == 0:
                loss = self._compute_loss(y, y_pred)
                self.losses.append(loss)

    def _compute_loss(self, y_true, y_pred):
        """Calcule la log loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict_proba(self, X):
        """Retourne les probabilités prédites"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Retourne les classes prédites"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
