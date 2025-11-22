import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Dataset:
    """Gestion et préparation des données patient"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_and_prepare(self, test_size=0.2, random_state=42):
        """Charge et prépare les données"""
        # Lecture du CSV
        df = pd.read_csv(self.filepath)

        # Nettoyage des colonnes (trim whitespace)
        df.columns = df.columns.str.strip()

        # Séparation features et target
        feature_cols = [
            "temperature",
            "frequence_cardiaque",
            "globules_blancs",
            "toux",
            "fatigue",
        ]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        # Conversion du statut en binaire (1 = Infecté, 0 = Sain)
        y = (df["statut"].str.strip() == "Infecté").astype(int).values

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Normalisation des features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return self

    def get_train_data(self):
        """Retourne les données d'entraînement"""
        return self.X_train, self.y_train

    def get_test_data(self):
        """Retourne les données de test"""
        return self.X_test, self.y_test

    def transform_patient(self, patient_data):
        """Transforme les données d'un patient unique pour la prédiction"""
        if isinstance(patient_data, dict):
            # Conversion dict vers array dans le bon ordre
            patient_array = np.array(
                [patient_data[feat] for feat in self.feature_names]
            ).reshape(1, -1)
        else:
            patient_array = np.array(patient_data).reshape(1, -1)

        return self.scaler.transform(patient_array)
