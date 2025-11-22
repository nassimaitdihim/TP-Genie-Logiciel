import numpy as np


class Evaluator:
    """Évaluation des performances du modèle"""

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def evaluate(self, on_test=True):
        """
        Évalue le modèle et retourne les métriques

        Args:
            on_test: Si True, évalue sur le test set, sinon sur le train set

        Returns:
            dict: dictionnaire contenant les métriques
        """
        if on_test:
            X, y_true = self.dataset.get_test_data()
            data_type = "Test"
        else:
            X, y_true = self.dataset.get_train_data()
            data_type = "Train"

        # Prédictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # Calcul des métriques
        metrics = self._compute_metrics(y_true, y_pred, y_proba)

        # Affichage
        print(f"\n{'='*50}")
        print(f"Métriques sur l'ensemble {data_type}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print(f"\nMatrice de confusion:")
        print(f"  TN: {metrics['tn']:3d}  FP: {metrics['fp']:3d}")
        print(f"  FN: {metrics['fn']:3d}  TP: {metrics['tp']:3d}")
        print(f"{'='*50}\n")

        return metrics

    def _compute_metrics(self, y_true, y_pred, y_proba):
        """Calcule toutes les métriques"""
        # Matrice de confusion
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Métriques
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # AUC (approximation simple)
        auc = self._compute_auc(y_true, y_proba)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc": auc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def _compute_auc(self, y_true, y_proba):
        """Calcule l'AUC-ROC"""
        # Tri par probabilités décroissantes
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # Calcul de l'AUC par la méthode des trapèzes
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp_cumsum = np.cumsum(y_true_sorted)
        fp_cumsum = np.cumsum(1 - y_true_sorted)

        tpr = tp_cumsum / n_pos
        fpr = fp_cumsum / n_neg

        # Calcul de l'aire sous la courbe
        auc = np.trapz(tpr, fpr)

        return auc
