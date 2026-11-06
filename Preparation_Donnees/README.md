# Préparation des données incendies

Le [notebook 04_Preparation_Donnees.ipynb](../Analyse_et_Modelisation/04_Preparation_Donnees.ipynb) présente le traitement, son code, les tableaux et les graphiques enregistrés.
Le [guide de prétraitement](GUIDE_PRETRAITEMENT.md) explique les décisions et l'interprétation des résultats.

## Traitement

- Contrôle des types, dates, valeurs manquantes, doublons et domaines physiques.
- Comparaison des imputations constante, moyenne, médiane, mode, KNN et régression sur des copies avec absences simulées.
- Diagnostics Z-score, IQR, boxplots, nuage de points, Isolation Forest, LOF et KNN.
- Comparaison sur copies du capping, du logarithme et de la racine carrée.
- Comparaison de MinMaxScaler, MaxAbsScaler, RobustScaler et StandardScaler.
- Encodage de la saison, de la cible et de catégories ordonnées ; caractéristiques calendaires cycliques.
- Découpage chronologique, validation interne, sélection de variables et PCA.
- Pipeline complet, matrices exportées, choix sur validation et évaluation finale.

Les 22 788 observations réelles sont conservées. Les simulations d'absence et de suppression restent distinctes des données utilisées dans le pipeline final.
Les valeurs météo extrêmes physiquement plausibles sont conservées.

## Reproduire

Utiliser l'environnement déclaré par `pyproject.toml` et `uv.lock` à la racine. Si nécessaire : `uv sync --locked`.
Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe .\Preparation_Donnees\pipeline_pretraitement.py
.\.venv\Scripts\python.exe .\Preparation_Donnees\verifier_resultats.py
```

Ouvrir le notebook avec le noyau Python du projet et lancer les cellules dans l'ordre.
Les sorties enregistrées permettent aussi sa lecture directe.
`executer_notebook.py` automatise cette exécution avec `jupyter_client`, fourni avec `ipykernel`.
`construire_notebook.py` reconstruit la version vierge du notebook et n'est pas nécessaire à son utilisation.

## Résultats

Les fichiers produits se trouvent dans `resultats/`.

| Fichier | Utilité |
| --- | --- |
| `rapport_application.json` | Contrôles, versions, empreinte source, choix et limites |
| `repartition_chronologique.csv` | 17 904 observations train, 2 374 validation, 2 510 test |
| `comparaison_validation.csv` | Comparaison des transformations sur 2024 |
| `evaluation_test_final.csv` | Évaluation figée sur 2025 et référence simple |
| `pipeline_complet.joblib` | Préparation et modèle réunis pour les entrées brutes |
| `X_*_pretraite.csv`, `y_*.csv` | Matrices et cibles identifiées par observation |
| `index_repartition.csv` | Dates, coordonnées et appartenance aux ensembles |
| Autres CSV et PNG | Comparaisons d'imputation, d'encodage, diagnostics et graphiques |
| `verification_resultats.json` | Vérification indépendante des exports et de l'inférence |

Dans les matrices CSV, `id_observation` sert uniquement à l'alignement. Il doit être retiré pour entraîner un autre modèle sur ces fichiers.
Le pipeline sauvegardé reçoit directement les sept colonnes brutes `lat_c`, `lon_c`, `acq_date`, `temperature_2m_max`, `precipitation_sum`, `wind_speed_10m_max` et `humidite_moy`.
Le module `pipeline_pretraitement` doit rester importable lors du rechargement, comme indiqué dans le guide et le notebook.

## Interprétation

Le protocole ajuste les paramètres sur 2018–2023, compare les variantes sur 2024, puis évalue le choix figé sur 2025.
La régression logistique permet d'examiner l'effet des transformations avec des réglages communs.
Le résultat dépend de ce jeu de données et ne définit pas une méthode de prétraitement universellement meilleure.
La météo est observée pendant la journée de la cible. L'expérience mesure une association rétrospective ; une prévision au début de la journée demanderait d'autres informations.
La constitution des exemples `feu=0`, le caractère forestier et la généralisation à de nouveaux lieux restent à documenter.

## Anciens résultats

Les sources et résultats de `Analyse_et_Modelisation/` sont conservés.
L'ancien notebook `03_Modelisation.ipynb` sauvegarde un scaler ajusté sur des entrées déjà standardisées. Son couple `modele_xgb_optimise.pkl` / `scaler.pkl` ne fournit donc pas une transformation correcte de nouvelles entrées brutes.
Pour réutiliser les résultats de ce dossier, employer son pipeline complet avec les entrées brutes attendues.
