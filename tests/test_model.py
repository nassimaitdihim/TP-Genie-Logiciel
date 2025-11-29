import pytest
import numpy as np
from core.model import ClinicalPredictor
from core.logistic_regression import LogisticRegression
from unittest.mock import Mock


class TestClinicalPredictor:
    """Tests pour la classe ClinicalPredictor"""

    @pytest.fixture
    def mock_model(self):
        """Crée un mock du modèle"""
        model = Mock(spec=LogisticRegression)
        model.predict_proba = Mock(return_value=np.array([0.7]))
        return model

    @pytest.fixture
    def mock_dataset(self):
        """Crée un mock du dataset"""
        dataset = Mock()
        dataset.transform_patient = Mock(return_value=np.array([[1, 2, 3, 4, 5]]))
        return dataset

    def test_initialization(self, mock_model):
        """Test l'initialisation du prédicteur"""
        predictor = ClinicalPredictor(mock_model)
        
        assert predictor.model == mock_model
        assert predictor.dataset is None

    def test_set_dataset(self, mock_model, mock_dataset):
        """Test l'association d'un dataset"""
        predictor = ClinicalPredictor(mock_model)
        result = predictor.set_dataset(mock_dataset)
        
        assert predictor.dataset == mock_dataset
        assert result == predictor  # Vérifier le chaînage

    def test_diagnose_infected(self, mock_model, mock_dataset):
        """Test le diagnostic d'un patient infecté"""
        mock_model.predict_proba = Mock(return_value=np.array([0.8]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 39.0,
            'frequence_cardiaque': 110,
            'globules_blancs': 14000,
            'toux': 1,
            'fatigue': 1
        }
        
        diagnosis = predictor.diagnose(patient_data)
        
        assert diagnosis == "Infecté"
        mock_dataset.transform_patient.assert_called_once()

    def test_diagnose_healthy(self, mock_model, mock_dataset):
        """Test le diagnostic d'un patient sain"""
        mock_model.predict_proba = Mock(return_value=np.array([0.3]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 36.8,
            'frequence_cardiaque': 70,
            'globules_blancs': 7000,
            'toux': 0,
            'fatigue': 0
        }
        
        diagnosis = predictor.diagnose(patient_data)
        
        assert diagnosis == "Sain"

    def test_diagnose_with_custom_threshold(self, mock_model, mock_dataset):
        """Test le diagnostic avec un seuil personnalisé"""
        mock_model.predict_proba = Mock(return_value=np.array([0.6]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 37.5,
            'frequence_cardiaque': 85,
            'globules_blancs': 9000,
            'toux': 1,
            'fatigue': 0
        }
        
        # Avec seuil par défaut (0.5)
        diagnosis_default = predictor.diagnose(patient_data, threshold=0.5)
        assert diagnosis_default == "Infecté"
        
        # Avec seuil élevé (0.7)
        diagnosis_high = predictor.diagnose(patient_data, threshold=0.7)
        assert diagnosis_high == "Sain"

    def test_diagnose_with_confidence(self, mock_model, mock_dataset):
        """Test le diagnostic avec niveau de confiance"""
        mock_model.predict_proba = Mock(return_value=np.array([0.85]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 39.5,
            'frequence_cardiaque': 115,
            'globules_blancs': 15000,
            'toux': 1,
            'fatigue': 1
        }
        
        diagnosis, confidence = predictor.diagnose_with_confidence(patient_data)
        
        assert diagnosis == "Infecté"
        assert confidence == 0.85
        assert 0 <= confidence <= 1

    def test_diagnose_without_dataset(self, mock_model):
        """Test le diagnostic sans dataset associé"""
        predictor = ClinicalPredictor(mock_model)
        
        # Données normalisées
        patient_data = np.array([0.5, 0.3, -0.2, 1, 0])
        
        diagnosis = predictor.diagnose(patient_data)
        
        # Devrait fonctionner avec des données déjà normalisées
        assert diagnosis in ["Infecté", "Sain"]

    def test_diagnose_array_input(self, mock_model, mock_dataset):
        """Test le diagnostic avec input array"""
        mock_model.predict_proba = Mock(return_value=np.array([0.7]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = [37.5, 85, 9000, 1, 0]
        
        diagnosis = predictor.diagnose(patient_data)
        
        assert diagnosis in ["Infecté", "Sain"]

    def test_boundary_case_exactly_threshold(self, mock_model, mock_dataset):
        """Test le cas limite où la probabilité est exactement au seuil"""
        mock_model.predict_proba = Mock(return_value=np.array([0.5]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 37.3,
            'frequence_cardiaque': 80,
            'globules_blancs': 8500,
            'toux': 0,
            'fatigue': 1
        }
        
        diagnosis = predictor.diagnose(patient_data, threshold=0.5)
        
        # Au seuil exact, devrait être classé comme "Infecté" (>=)
        assert diagnosis == "Infecté"

    def test_very_high_confidence(self, mock_model, mock_dataset):
        """Test avec une très haute confiance"""
        mock_model.predict_proba = Mock(return_value=np.array([0.99]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 40.5,
            'frequence_cardiaque': 130,
            'globules_blancs': 18000,
            'toux': 1,
            'fatigue': 1
        }
        
        diagnosis, confidence = predictor.diagnose_with_confidence(patient_data)
        
        assert diagnosis == "Infecté"
        assert confidence >= 0.95

    def test_very_low_confidence(self, mock_model, mock_dataset):
        """Test avec une très basse confiance"""
        mock_model.predict_proba = Mock(return_value=np.array([0.01]))
        
        predictor = ClinicalPredictor(mock_model)
        predictor.set_dataset(mock_dataset)
        
        patient_data = {
            'temperature': 36.2,
            'frequence_cardiaque': 60,
            'globules_blancs': 5500,
            'toux': 0,
            'fatigue': 0
        }
        
        diagnosis, confidence = predictor.diagnose_with_confidence(patient_data)
        
        assert diagnosis == "Sain"
        assert confidence <= 0.05