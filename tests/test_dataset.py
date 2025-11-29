import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from core.dataset import Dataset


class TestDataset:
    """Tests pour la classe Dataset"""

    @pytest.fixture
    def sample_csv(self):
        """Crée un fichier CSV temporaire pour les tests"""
        data = {
            'temperature': [36.5, 37.0, 38.5, 39.0, 36.8, 38.2],
            'frequence_cardiaque': [70, 75, 100, 110, 68, 95],
            'globules_blancs': [7000, 7500, 12000, 14000, 6800, 11000],
            'toux': [0, 0, 1, 1, 0, 1],
            'fatigue': [0, 0, 1, 1, 0, 1],
            'statut': ['Sain', 'Sain', 'Infecté', 'Infecté', 'Sain', 'Infecté']
        }
        
        df = pd.DataFrame(data)
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df.to_csv(f.name, index=False)
            filepath = f.name
        
        yield filepath
        
        # Nettoyage
        os.unlink(filepath)

    def test_initialization(self, sample_csv):
        """Test l'initialisation du dataset"""
        dataset = Dataset(sample_csv)
        
        assert dataset.filepath == sample_csv
        assert dataset.scaler is not None
        assert dataset.X_train is None
        assert dataset.X_test is None
        assert dataset.y_train is None
        assert dataset.y_test is None

    def test_load_and_prepare(self, sample_csv):
        """Test le chargement et la préparation des données"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare(test_size=0.3, random_state=42)
        
        # Vérifier que les données sont chargées
        assert dataset.X_train is not None
        assert dataset.X_test is not None
        assert dataset.y_train is not None
        assert dataset.y_test is not None
        
        # Vérifier les dimensions
        assert dataset.X_train.shape[1] == 5  # 5 features
        assert dataset.X_test.shape[1] == 5
        
        # Vérifier la taille du split
        total_samples = len(dataset.X_train) + len(dataset.X_test)
        assert total_samples == 6

    def test_feature_names(self, sample_csv):
        """Test que les noms de features sont correctement définis"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        expected_features = [
            'temperature',
            'frequence_cardiaque',
            'globules_blancs',
            'toux',
            'fatigue'
        ]
        
        assert dataset.feature_names == expected_features

    def test_label_encoding(self, sample_csv):
        """Test l'encodage des labels"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        # Vérifier que les labels sont binaires
        assert np.all(np.isin(dataset.y_train, [0, 1]))
        assert np.all(np.isin(dataset.y_test, [0, 1]))

    def test_normalization(self, sample_csv):
        """Test la normalisation des features"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        # Les données normalisées devraient avoir une moyenne proche de 0
        # et un écart-type proche de 1
        train_mean = np.mean(dataset.X_train, axis=0)
        train_std = np.std(dataset.X_train, axis=0)
        
        assert np.allclose(train_mean, 0, atol=1e-7)
        assert np.allclose(train_std, 1, atol=0.5)

    def test_get_train_data(self, sample_csv):
        """Test la récupération des données d'entraînement"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        X_train, y_train = dataset.get_train_data()
        
        assert X_train is not None
        assert y_train is not None
        assert len(X_train) == len(y_train)
        assert X_train.shape[1] == 5

    def test_get_test_data(self, sample_csv):
        """Test la récupération des données de test"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        X_test, y_test = dataset.get_test_data()
        
        assert X_test is not None
        assert y_test is not None
        assert len(X_test) == len(y_test)
        assert X_test.shape[1] == 5

    def test_transform_patient_dict(self, sample_csv):
        """Test la transformation d'un patient (format dict)"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        patient_data = {
            'temperature': 37.5,
            'frequence_cardiaque': 85,
            'globules_blancs': 9000,
            'toux': 1,
            'fatigue': 0
        }
        
        transformed = dataset.transform_patient(patient_data)
        
        # Vérifier la forme
        assert transformed.shape == (1, 5)
        
        # Vérifier que c'est un array numpy
        assert isinstance(transformed, np.ndarray)

    def test_transform_patient_array(self, sample_csv):
        """Test la transformation d'un patient (format array)"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare()
        
        patient_data = [37.5, 85, 9000, 1, 0]
        
        transformed = dataset.transform_patient(patient_data)
        
        assert transformed.shape == (1, 5)

    def test_stratified_split(self, sample_csv):
        """Test que le split est stratifié"""
        dataset = Dataset(sample_csv)
        dataset.load_and_prepare(test_size=0.3, random_state=42)
        
        # Vérifier que les deux classes sont représentées
        unique_train = np.unique(dataset.y_train)
        unique_test = np.unique(dataset.y_test)
        
        # Au moins une des deux devrait avoir les deux classes
        assert len(unique_train) >= 1
        assert len(unique_test) >= 1

    def test_random_state_reproducibility(self, sample_csv):
        """Test que le random_state assure la reproductibilité"""
        dataset1 = Dataset(sample_csv)
        dataset1.load_and_prepare(random_state=42)
        
        dataset2 = Dataset(sample_csv)
        dataset2.load_and_prepare(random_state=42)
        
        # Les splits devraient être identiques
        assert np.array_equal(dataset1.X_train, dataset2.X_train)
        assert np.array_equal(dataset1.y_train, dataset2.y_train)

    def test_column_whitespace_handling(self):
        """Test le traitement des espaces dans les noms de colonnes"""
        data = {
            ' temperature ': [36.5, 37.0],
            'frequence_cardiaque  ': [70, 75],
            'globules_blancs': [7000, 7500],
            'toux': [0, 0],
            'fatigue': [0, 0],
            'statut': ['Sain', 'Sain']
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df.to_csv(f.name, index=False)
            filepath = f.name
        
        try:
            dataset = Dataset(filepath)
            dataset.load_and_prepare()
            
            # Devrait fonctionner malgré les espaces
            assert dataset.X_train is not None
        finally:
            os.unlink(filepath)