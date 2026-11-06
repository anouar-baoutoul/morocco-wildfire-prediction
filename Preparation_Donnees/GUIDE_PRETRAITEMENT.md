# Préparation des données de détection des incendies au Maroc

Ce guide décrit les choix de préparation du jeu de données, les comparaisons réalisées et les limites des résultats.
Il accompagne [04_Preparation_Donnees.ipynb](../Analyse_et_Modelisation/04_Preparation_Donnees.ipynb) et le script [pipeline_pretraitement.py](pipeline_pretraitement.py).
Les tableaux, figures et objets réutilisables sont enregistrés dans `resultats/`.
La source est conservée dans `Analyse_et_Modelisation/data/dataset_feux_meteo_complete.csv`, depuis la racine du projet.

## Organisation du traitement

1. **Nettoyage** : contrôler types, dates, valeurs absentes, doublons, unités et cohérence des observations.
2. **Choix des variables** : retenir des entrées pertinentes et écarter cible, identifiants et informations d'audit des prédicteurs.
3. **Transformation** : comparer les mises à l'échelle et les changements de distribution.
4. **Construction de caractéristiques** : dériver la saison et une représentation cyclique du jour de l'année.
5. **Analyse des représentations** : examiner une sélection statistique et une réduction de dimension sur l'entraînement.

Les transformations apprises sont ajustées exclusivement sur l'ensemble d'entraînement. Les comparaisons utilisent une validation distincte, puis le choix figé est évalué sur le test.

## Contrôles de qualité

Le fichier source comporte 22 788 observations. Le contrôle du fichier conservé ne relève aucune valeur manquante ni aucun doublon exact.
Les vérifications portent sur le schéma, les types numériques, les dates, la cible binaire et les domaines physiques.
Les dates de détection et de météo doivent concorder. Le mois et le jour de l'année doivent correspondre à la date d'acquisition.
Une humidité hors de [0, 100], une pluie négative, un vent négatif ou une date incohérente provoquent un arrêt pour examen.
Des observations contradictoires pour un même lieu et un même jour demandent également une vérification.
L'identifiant d'observation assure la traçabilité et l'alignement des exports ; il ne constitue pas une variable du modèle.
Les coordonnées de grille et les autres informations d'audit ne sont pas ajoutées automatiquement aux prédicteurs.
L'empreinte SHA-256 du fichier source permet d'identifier précisément la version utilisée.

## Données manquantes et simulation d'absences

Le diagnostic mesure le nombre de NaN et leur proportion par colonne. L'absence actuelle de NaN décrit le fichier reçu, sans prouver que toutes les mesures aient été disponibles pendant la collecte initiale.

- **MCAR** : l'absence est indépendante des valeurs observées et non observées.
- **MAR** : après conditionnement sur les variables observées pertinentes, l'absence ne dépend plus de la valeur non observée.
- **MNAR** : une dépendance à la valeur non observée subsiste après prise en compte des informations disponibles.

Un taux d'absence ne permet pas, à lui seul, d'identifier ces mécanismes. La connaissance de la collecte et des hypothèses explicites restent nécessaires.
Des seuils tels que 5 % de lignes ou 50 % d'absences dans une colonne sont des repères à discuter. La perte d'information et le biais possible comptent autant que le pourcentage.

| Méthode | Intérêt | Limite |
| --- | --- | --- |
| Suppression de lignes | Traitement simple d'observations dispensables | Perte d'effectif et risque de biais |
| Suppression de colonnes | Écarte un attribut très incomplet ou peu utile | Peut supprimer une information importante |
| Constante ou catégorie « Inconnu » | Rend l'absence explicite | Zéro peut avoir une signification physique |
| Moyenne | Repère pour une variable quantitative assez symétrique | Sensibilité aux extrêmes, variance réduite |
| Médiane | Moins sensible aux extrêmes | Ignore les relations entre variables |
| Mode | Adapté aux catégories | Renforce la catégorie majoritaire |
| KNN | Utilise la proximité entre observations | Dépend des échelles et des voisins disponibles |
| Régression itérative | Exploite les relations entre variables | Peut produire des valeurs physiquement peu plausibles |

La comparaison d'imputation prélève au maximum 1 200 lignes complètes du train. Elle réserve 20 % de cet échantillon à une évaluation interne et masque artificiellement environ 5 % des cellules météo par variable, avec la graine 42.
Les imputateurs sont ajustés sur les donneurs masqués des 80 % restants. Les vérités masquées réservées ne participent jamais au fit.
KNN et la régression itérative utilisent une standardisation ajustée sur ces donneurs uniquement.
Les MAE et RMSE portent sur les cellules masquées réservées, par variable et dans son unité initiale. Une RMSE en degrés ne se compare pas directement à une RMSE en millimètres.
L'évaluation porte ici sur 12 cellules masquées par variable : elle montre les comportements des méthodes sans établir un classement général fiable.
Les essais ne modifient pas les observations réelles et la cible n'est jamais imputée.

## Valeurs atypiques et météo extrême

Une température élevée ou un vent fort peut constituer une information utile pour l'étude des incendies. Une valeur rare ne suffit donc pas à justifier une suppression.
Une correction de mesure exige une source fiable ; aucune valeur de remplacement n'est inventée.

- **Z-score** : `z = (x − moyenne_train) / écart_type_train`, avec `ddof=0`. Le diagnostic utilise `|z| > 3` et montre une zone de vigilance entre 2 et 3.
- **IQR** : `IQR = Q3 − Q1`, bornes `Q1 − 1,5 × IQR` et `Q3 + 1,5 × IQR`.
- **Graphiques** : boxplots, histogrammes et nuage température/humidité rendent visibles asymétrie, amas et relations entre variables.

Pour la pluie du train, `Q1 = Q3 = 0`. L'IQR vaut zéro et toute pluie positive peut être signalée. Un écrêtage mécanique à cette borne effacerait la pluie observée.
Les valeurs réelles sont conservées. Les suppressions hypothétiques, le capping, `log1p(x)` et la racine carrée sont comparés sur des copies.
Le logarithme et la racine s'appliquent ici aux variables non négatives. Ces opérations changent l'interprétation des valeurs et ne répondent pas au même objectif qu'un changement d'échelle linéaire.
Isolation Forest, LOF et la distance KNN complètent le diagnostic multivarié sur un échantillon du train. Le seuil KNN au quantile 95 est un paramètre exploratoire ; les signaux ne constituent pas une vérité terrain.

## Mise à l'échelle

Pour une variable non constante, tous les paramètres des formules suivantes proviennent du train.

| Méthode | Formule usuelle | Propriété |
| --- | --- | --- |
| MinMaxScaler | `(x − min_train) / (max_train − min_train)` | Train dans [0, 1], sensible aux extrêmes |
| MaxAbsScaler | `x / max_train(abs(x))` | Train dans [−1, 1], zéros conservés |
| RobustScaler | `(x − médiane_train) / (Q3_train − Q1_train)` | Centrage robuste, sans intervalle garanti |
| StandardScaler | `(x − moyenne_train) / écart_type_train` | Moyenne 0 et écart-type 1 sur le train |

RobustScaler centre sur la médiane et utilise l'IQR, conformément à la [documentation officielle](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).
StandardScaler utilise `ddof=0` : `écart_type = sqrt(somme((x − moyenne)²) / n)`. Avec pandas, vérifier avec `std(ddof=0)` ; `std()` utilise habituellement `ddof=1`. Voir la [documentation StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
La standardisation ne rend pas une distribution gaussienne. Les valeurs changent d'unité et s'expriment en écarts-types.
Validation et test réutilisent les paramètres du train ; leur moyenne n'a donc pas à être nulle et leur écart-type n'a pas à être unitaire.
Une valeur future supérieure au maximum du train peut dépasser 1 après MinMax. Le comportement par défaut ne la tronque pas. Une sortie de l'intervalle d'apprentissage est aussi possible avec MaxAbs. Voir [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
Une dispersion nulle fait l'objet d'une protection contre la division par zéro dans scikit-learn.
Les quatre scalers sont comparés séparément avec un protocole commun.

## Encodage et caractéristiques calendaires

La saison est dérivée du mois : hiver de décembre à février, printemps de mars à mai, été de juin à août, automne de septembre à novembre.
OneHotEncoder produit des colonnes binaires sans imposer de distance numérique entre les saisons. La comparaison d'encodage affiche quatre colonnes ; le pipeline logistique utilise `drop="first"` et conserve trois colonnes, avec une saison de référence.
Le fit utilise le train. `handle_unknown="ignore"` transforme une catégorie non apprise en vecteur de zéros, sans nouvel ajustement sur la validation.
LabelEncoder porte sur une copie textuelle de la cible, avec les classes `absence_detection` et `detection`. La cible réelle `feu` reste codée 0/1.
Cet encodeur concerne les étiquettes cibles. Un ordre métier dans les entrées doit être fourni à OrdinalEncoder. Voir la [documentation LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
La comparaison ordinale définit `faible < moyenne < élevée` pour une humidité ≤ 30 %, entre 30 % et 60 %, puis > 60 %. Ces seuils servent à examiner l'encodage ; ils ne constituent pas une classification validée du risque et ne remplacent pas l'humidité continue dans le modèle final.
Deux coordonnées cycliques, sinus et cosinus du jour de l'année, représentent la proximité entre fin décembre et début janvier. Leur période tient compte des années bissextiles. Ces dérivations utilisent uniquement la date et aucune étiquette.

## Répartition et prévention des fuites

| Ensemble | Période | Observations | Rôle |
| --- | --- | --- | --- |
| Entraînement | 2018–2023 | 17 904 | Ajustement des paramètres |
| Validation | 2024 | 2 374 | Comparaison des variantes |
| Test | 2025 | 2 510 | Évaluation finale du choix figé |

Les données sont triées par date et les identifiants des trois ensembles sont disjoints.
Tous les paramètres appris respectent cette séparation : moyenne, médiane, voisins, scaler, catégories, sélection, PCA et modèle.
Un pipeline réajuste ses étapes dans chaque pli lorsqu'il est utilisé en validation croisée.
Une comparaison interne utilise `train_test_split(test_size=0.2, random_state=42, stratify=y)` à l'intérieur du train. Elle contrôle la reproductibilité et la conservation approximative des proportions de classes.
Un k-fold stratifié interne est également présenté. Ces partitions aléatoires ne remplacent pas le découpage temporel principal.
La validation chronologique interne utilise successivement 2022 et 2023 comme années de contrôle, avec les années antérieures pour ajuster chaque pipeline.
SelectKBest et PCA sont examinés sur le seul train préparé. La PCA conserve 95 % de la variance et peut réduire l'interprétabilité.
Un score univarié élevé ou une variance expliquée élevée ne prouvent pas une amélioration de la prédiction. Ces analyses complémentaires ne changent pas le pipeline final.

## Comparaison et évaluation

Sept configurations sont comparées avec une régression logistique : données sans mise à l'échelle, Standard, MinMax, MaxAbs, Robust, puis Standard et Robust avec `log1p(pluie)`.
Chaque configuration est ajustée sur 2018–2023 et évaluée sur 2024. Le critère de choix est l'average precision, puis la ROC-AUC pour départager.
La comparaison utilise le même modèle et ses mêmes réglages. Les résultats dépendent de l'échantillon, de la transformation et de la régularisation ; ils ne désignent pas un scaler universellement meilleur.
Le seuil de classification reste fixé à 0,5. Le choix figé est évalué sur 2025 avec une référence prédisant la fréquence du train.

## Exécution et réutilisation

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe .\Preparation_Donnees\pipeline_pretraitement.py
.\.venv\Scripts\python.exe .\Preparation_Donnees\verifier_resultats.py
```

Le notebook peut être ouvert avec le noyau Python du projet et exécuté dans l'ordre.
Les comparaisons de `comparaisons.py` et les diagnostics de `aberrants.py` complètent le script principal.
Le fichier `pipeline_complet.joblib` réunit préparation et modèle. Le module `pipeline_pretraitement.py` doit rester accessible à Python pour recharger sa fonction de création des caractéristiques.

```python
from pathlib import Path
import sys
import joblib

dossier = Path("Preparation_Donnees").resolve()
sys.path.insert(0, str(dossier))
import pipeline_pretraitement

pipeline = joblib.load(dossier / "resultats" / "pipeline_complet.joblib")
```

Le pipeline reçoit les sept colonnes brutes `lat_c`, `lon_c`, `acq_date`, `temperature_2m_max`, `precipitation_sum`, `wind_speed_10m_max` et `humidite_moy`, dans leurs unités d'origine.
Les transformations sauvegardées sont réutilisées directement. Un nouveau scaler ne doit pas être ajusté sur les observations à prédire.
Dans les matrices CSV exportées, `id_observation` sert uniquement à l'alignement et doit être retiré avant l'entraînement d'un autre modèle.

## Portée des résultats

La comparaison mesure l'effet des prétraitements sur ce jeu de données. Ses métriques ne constituent pas une validation opérationnelle du risque d'incendie au Maroc.
La météo décrit la même journée que la cible. Les maxima, cumuls et moyennes journaliers ne sont pas connus au début de cette journée : les résultats relèvent d'une association rétrospective.
La construction des exemples `feu=0` reste à documenter. Une absence de détection satellitaire ne certifie pas l'absence réelle d'incendie et une détection thermique ne certifie pas le caractère forestier sans vérification de l'occupation du sol.
Les coordonnées arrondies et les points proches des limites restent soumis aux réserves du contrôle géographique existant.
Un même lieu peut apparaître sur plusieurs années : ce test ne mesure donc pas la généralisation à de nouveaux lieux. Les probabilités ne sont pas calibrées sur la prévalence réelle dans la population.
Pour étudier une prévision à l'avance, il faudra un horizon explicite, des informations disponibles avant cet horizon, des séries quotidiennes complètes et une validation indépendante. Un décalage arbitraire de lignes ne remplace pas ces éléments.
