import numpy as np
import pandas as pd

def generate_clinical_data_csv(filename='clinical_data.csv', n_samples=500):
    """
    Génère un fichier CSV avec des données cliniques simulées
    
    Args:
        filename: nom du fichier CSV à créer
        n_samples: nombre total d'échantillons à générer
    """
    np.random.seed(42)
    
    # Patients sains (label = 0)
    n_healthy = n_samples // 2
    print(f"Génération de {n_healthy} patients sains...")
    
    healthy_data = {
        'temperature': np.random.normal(36.8, 0.3, n_healthy),
        'frequence_cardiaque': np.random.normal(70, 8, n_healthy),
        'globules_blancs': np.random.normal(7000, 1000, n_healthy),
        'toux': np.random.choice([0, 1], n_healthy, p=[0.9, 0.1]),
        'fatigue': np.random.choice([0, 1], n_healthy, p=[0.8, 0.2]),
        'statut': ['Sain'] * n_healthy
    }
    
    # Patients infectés (label = 1)
    n_infected = n_samples - n_healthy
    print(f"Génération de {n_infected} patients infectés...")
    
    infected_data = {
        'temperature': np.random.normal(38.5, 0.8, n_infected),
        'frequence_cardiaque': np.random.normal(95, 12, n_infected),
        'globules_blancs': np.random.normal(12000, 2000, n_infected),
        'toux': np.random.choice([0, 1], n_infected, p=[0.2, 0.8]),
        'fatigue': np.random.choice([0, 1], n_infected, p=[0.1, 0.9]),
        'statut': ['Infecté'] * n_infected
    }
    
    # Créer des DataFrames
    df_healthy = pd.DataFrame(healthy_data)
    df_infected = pd.DataFrame(infected_data)
    
    # Combiner les deux DataFrames
    df = pd.concat([df_healthy, df_infected], ignore_index=True)
    
    # Mélanger les données
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Arrondir les valeurs numériques pour plus de lisibilité
    df['temperature'] = df['temperature'].round(1)
    df['frequence_cardiaque'] = df['frequence_cardiaque'].round(0).astype(int)
    df['globules_blancs'] = df['globules_blancs'].round(0).astype(int)
    
    # Sauvegarder en CSV
    df.to_csv(filename, index=False)
    
    print(f"\n✓ Fichier '{filename}' créé avec succès!")
    print(f"  Total d'échantillons: {len(df)}")
    print(f"  - Patients sains: {len(df[df['statut'] == 'Sain'])}")
    print(f"  - Patients infectés: {len(df[df['statut'] == 'Infecté'])}")
    
    # Afficher les premières lignes
    print(f"\nAperçu des données:")
    print(df.head(10))
    
    # Statistiques
    print(f"\nStatistiques descriptives:")
    print(df.groupby('statut').agg({
        'temperature': ['mean', 'std'],
        'frequence_cardiaque': ['mean', 'std'],
        'globules_blancs': ['mean', 'std'],
        'toux': 'sum',
        'fatigue': 'sum'
    }).round(2))
    
    return df


if __name__ == "__main__":
    # Générer le fichier CSV
    df = generate_clinical_data_csv('clinical_data.csv', n_samples=500)
    
    print("\n" + "="*60)
    print("Le fichier CSV est prêt à être utilisé avec ClinicalPredictor!")
    print("="*60)