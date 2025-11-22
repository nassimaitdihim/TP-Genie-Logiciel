import sys

sys.path.append("..")
from core.dataset import Dataset
from core.model import ClinicalPredictor
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator


def main():
    print("=" * 60)
    print("SYSTÈME DE PRÉDICTION CLINIQUE")
    print("=" * 60)

    print("\n[1/4] Chargement des données...")
    dataset = Dataset("../data/patient_data.csv")
    dataset.load_and_prepare(test_size=0.2, random_state=42)
    print(
        f"✓ Données chargées: {len(dataset.X_train)} train, {len(dataset.X_test)} test"
    )

    print("\n[2/4] Entraînement du modèle...")
    trainer = Trainer(dataset)
    model = trainer.train(
        model_params={
            "learning_rate": 0.1,
            "n_iterations": 2000,
            "regularization": 0.01,
        }
    )
    print(f"✓ Modèle entraîné avec {len(trainer.get_training_history())} checkpoints")

    print("\n[3/4] Évaluation du modèle...")
    evaluator = Evaluator(model, dataset)
    train_metrics = evaluator.evaluate(on_test=False)
    test_metrics = evaluator.evaluate(on_test=True)

    print("\n[4/4] Initialisation du prédicteur clinique...")
    predictor = ClinicalPredictor(model)
    predictor.set_dataset(dataset)
    print("✓ ClinicalPredictor prêt à l'emploi!")

    # Démonstration avec quelques exemples
    print("\n" + "=" * 60)
    print("EXEMPLES DE PRÉDICTIONS")
    print("=" * 60)

    # Exemple 1: Patient probablement infecté
    patient_infected = {
        "temperature": 39.5,
        "frequence_cardiaque": 110,
        "globules_blancs": 14500,
        "toux": 1,
        "fatigue": 1,
    }

    diagnosis, confidence = predictor.diagnose_with_confidence(patient_infected)
    print(f"\nPatient 1 (symptômes d'infection):")
    print(f"  Température: {patient_infected['temperature']}°C")
    print(f"  Fréquence cardiaque: {patient_infected['frequence_cardiaque']} bpm")
    print(f"  Globules blancs: {patient_infected['globules_blancs']}")
    print(f"  Toux: {'Oui' if patient_infected['toux'] else 'Non'}")
    print(f"  Fatigue: {'Oui' if patient_infected['fatigue'] else 'Non'}")
    print(f"  → Diagnostic: {diagnosis} (confiance: {confidence:.2%})")

    # Exemple 2: Patient probablement sain
    patient_healthy = {
        "temperature": 36.8,
        "frequence_cardiaque": 72,
        "globules_blancs": 6500,
        "toux": 0,
        "fatigue": 0,
    }

    diagnosis, confidence = predictor.diagnose_with_confidence(patient_healthy)
    print(f"\nPatient 2 (paramètres normaux):")
    print(f"  Température: {patient_healthy['temperature']}°C")
    print(f"  Fréquence cardiaque: {patient_healthy['frequence_cardiaque']} bpm")
    print(f"  Globules blancs: {patient_healthy['globules_blancs']}")
    print(f"  Toux: {'Oui' if patient_healthy['toux'] else 'Non'}")
    print(f"  Fatigue: {'Oui' if patient_healthy['fatigue'] else 'Non'}")
    print(f"  → Diagnostic: {diagnosis} (confiance: {confidence:.2%})")

    # Exemple 3: Cas limite
    patient_borderline = {
        "temperature": 37.6,
        "frequence_cardiaque": 88,
        "globules_blancs": 9800,
        "toux": 1,
        "fatigue": 0,
    }

    diagnosis, confidence = predictor.diagnose_with_confidence(patient_borderline)
    print(f"\nPatient 3 (cas limite):")
    print(f"  Température: {patient_borderline['temperature']}°C")
    print(f"  Fréquence cardiaque: {patient_borderline['frequence_cardiaque']} bpm")
    print(f"  Globules blancs: {patient_borderline['globules_blancs']}")
    print(f"  Toux: {'Oui' if patient_borderline['toux'] else 'Non'}")
    print(f"  Fatigue: {'Oui' if patient_borderline['fatigue'] else 'Non'}")
    print(f"  → Diagnostic: {diagnosis} (confiance: {confidence:.2%})")

    print("\n" + "=" * 60)
    print("Pipeline terminé avec succès!")
    print("=" * 60)

    return predictor, model, dataset


if __name__ == "__main__":
    predictor, model, dataset = main()
