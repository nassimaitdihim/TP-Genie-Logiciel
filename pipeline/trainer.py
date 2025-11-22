import sys

sys.path.append("..")

from core.logistic_regression import LogisticRegression


class Trainer:
    """Gestionnaire d'entraînement des modèles"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def train(self, model_params=None):
        """
        Entraîne un modèle de régression logistique

        Args:
            model_params: dict contenant les hyperparamètres du modèle
                         {
                             'learning_rate': float,
                             'n_iterations': int,
                             'regularization': float
                         }

        Returns:
            LogisticRegression: modèle entraîné
        """
        if model_params is None:
            model_params = {
                "learning_rate": 0.1,
                "n_iterations": 2000,
                "regularization": 0.01,
            }

        # Initialisation du modèle
        self.model = LogisticRegression(**model_params)

        # Récupération des données d'entraînement
        X_train, y_train = self.dataset.get_train_data()

        # Entraînement
        print(f"Entraînement du modèle avec {len(X_train)} échantillons...")
        self.model.fit(X_train, y_train)
        print("Entraînement terminé!")

        return self.model

    def get_model(self):
        """Retourne le modèle entraîné"""
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas encore été entraîné. Appelez train() d'abord."
            )
        return self.model

    def get_training_history(self):
        """Retourne l'historique des losses pendant l'entraînement"""
        if self.model is None:
            return []
        return self.model.losses
