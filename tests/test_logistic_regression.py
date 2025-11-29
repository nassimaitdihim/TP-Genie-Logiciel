import pytest
import numpy as np
from core.logistic_regression import LogisticRegression


class TestLogisticRegression:
    """Tests pour la classe LogisticRegression"""

    def test_initialization(self):
        """Test l'initialisation du modèle"""
        model = LogisticRegression(learning_rate=0.01, n_iterations=100, regularization=0.1)
        assert model.learning_rate == 0.01
        assert model.n_iterations == 100
        assert model.regularization == 0.1
        assert model.weights is None
        assert model.bias is None
        assert model.losses == []

    def test_sigmoid(self):
        """Test la fonction sigmoid"""
        model = LogisticRegression()
        
        # Test valeurs typiques
        assert np.isclose(model._sigmoid(0), 0.5)
        assert np.isclose(model._sigmoid(10), 0.9999546, atol=1e-5)
        assert np.isclose(model._sigmoid(-10), 0.0000454, atol=1e-5)
        
        # Test avec array
        z = np.array([0, 1, -1])
        result = model._sigmoid(z)
        assert result.shape == (3,)
        assert np.isclose(result[0], 0.5)

    def test_fit_simple_dataset(self):
        """Test l'entraînement sur un dataset simple"""
        # Dataset linéairement séparable
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)
        
        # Vérifier que les poids ont été initialisés
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.weights) == 2
        
        # Vérifier que les losses ont été enregistrées
        assert len(model.losses) > 0
        
        # Vérifier que la loss diminue
        assert model.losses[-1] < model.losses[0]

    def test_predict_proba(self):
        """Test la prédiction de probabilités"""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        
        probas = model.predict_proba(X)
        
        # Vérifier la forme
        assert probas.shape == (4,)
        
        # Vérifier que les probabilités sont entre 0 et 1
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

    def test_predict(self):
        """Test la prédiction de classes"""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Vérifier la forme
        assert predictions.shape == (4,)
        
        # Vérifier que les prédictions sont binaires
        assert np.all(np.isin(predictions, [0, 1]))

    def test_predict_with_threshold(self):
        """Test la prédiction avec différents seuils"""
        X = np.array([[1, 2], [2, 3]])
        y = np.array([0, 1])
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Test avec différents seuils
        pred_05 = model.predict(X, threshold=0.5)
        pred_03 = model.predict(X, threshold=0.3)
        pred_07 = model.predict(X, threshold=0.7)
        
        assert pred_05.shape == (2,)
        assert pred_03.shape == (2,)
        assert pred_07.shape == (2,)

    def test_compute_loss(self):
        """Test le calcul de la loss"""
        model = LogisticRegression()
        
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        
        loss = model._compute_loss(y_true, y_pred)
        
        assert isinstance(loss, float)
        assert loss > 0

    def test_regularization_effect(self):
        """Test l'effet de la régularisation"""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        # Modèle sans régularisation
        model_no_reg = LogisticRegression(regularization=0, n_iterations=500)
        model_no_reg.fit(X, y)
        
        # Modèle avec régularisation
        model_reg = LogisticRegression(regularization=1.0, n_iterations=500)
        model_reg.fit(X, y)
        
        # Les poids devraient être différents
        assert not np.allclose(model_no_reg.weights, model_reg.weights)

    def test_fit_with_single_feature(self):
        """Test l'entraînement avec une seule feature"""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        model = LogisticRegression(n_iterations=500)
        model.fit(X, y)
        
        assert model.weights.shape == (1,)
        assert model.bias is not None

    def test_perfect_separation(self):
        """Test avec des données parfaitement séparables"""
        X = np.array([[1], [2], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        # Devrait prédire parfaitement
        accuracy = np.mean(predictions == y)
        assert accuracy >= 0.75  # Au moins 75% de précision