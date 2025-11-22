class ClinicalPredictor:
    """Prédicteur clinique pour diagnostiquer l'état d'infection des patients"""

    def __init__(self, model):
        """
        Initialise le prédicteur avec un modèle d'apprentissage automatique

        Args:
            model: Modèle ML déjà entraîné (doit avoir une méthode predict_proba)
        """
        self.model = model
        self.dataset = None

    def set_dataset(self, dataset):
        """Associe un dataset pour la transformation des données"""
        self.dataset = dataset
        return self

    def diagnose(self, patient_data, threshold=0.5):
        """
        Effectue un diagnostic sur les données d'un patient

        Args:
            patient_data: dict ou array contenant les caractéristiques du patient
                         Format dict: {
                             'temperature': float,
                             'frequence_cardiaque': float,
                             'globules_blancs': float,
                             'toux': int (0 ou 1),
                             'fatigue': int (0 ou 1)
                         }
            threshold: seuil de décision (défaut: 0.5)

        Returns:
            str: "Infecté" ou "Sain"
        """
        # Transformation des données patient
        if self.dataset is not None:
            X = self.dataset.transform_patient(patient_data)
        else:
            # Si pas de dataset, on suppose que les données sont déjà normalisées
            import numpy as np

            X = np.array(patient_data).reshape(1, -1)

        # Prédiction de la probabilité
        proba = self.model.predict_proba(X)[0]

        # Retour du diagnostic basé sur le seuil
        return "Infecté" if proba >= threshold else "Sain"

    def diagnose_with_confidence(self, patient_data):
        """
        Effectue un diagnostic avec le niveau de confiance

        Returns:
            tuple: (diagnostic, probabilité)
        """
        if self.dataset is not None:
            X = self.dataset.transform_patient(patient_data)
        else:
            import numpy as np

            X = np.array(patient_data).reshape(1, -1)

        proba = self.model.predict_proba(X)[0]
        diagnosis = "Infecté" if proba >= 0.5 else "Sain"

        return diagnosis, proba
