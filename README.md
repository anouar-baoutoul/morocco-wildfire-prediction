# 🔥 Prédiction des feux de forêt au Maroc

Projet réalisé par **Adam TALIBI et Anouar BAOUTOUL** à l’**EMSI**, dans le cadre de l’année universitaire **2026/2027**.

## 🎯 Objectif du projet

L’objectif est d’étudier des observations de feux de forêt au Maroc ainsi que les conditions météorologiques associées afin de préparer un modèle de **classification permettant de prédire la présence ou l’absence d’un feu**.

L’estimation de la **surface brûlée** pourra également être étudiée ultérieurement, mais elle nécessite une variable cible supplémentaire qui n’est pas disponible dans la base actuelle.

---

## 📊 Données utilisées

Le jeu de données préparé contient :

- **22 788 observations**
- Données couvrant la période **2018–2025**
- **1 232 coordonnées géographiques distinctes**
- **13 colonnes dans le CSV source**, puis **17 colonnes dans la table préparée** avec les informations de traçabilité
- **11 variables après transformation** dans le pipeline de classification final
- Aucune valeur manquante après préparation

La variable cible est :

```text
feu
```

Répartition :

- `feu = 0` : **14 761 observations**
- `feu = 1` : **8 027 observations**

Ces proportions correspondent uniquement à notre échantillon et ne représentent pas directement le risque réel d’incendie au Maroc.

---

# 📌 Étapes du projet

## 1. Collecte et complément des données météo

Le projet utilise le fichier CSV fourni comme source initiale.

La provenance déclarée des observations de feux est **NASA FIRMS**.

Les informations météorologiques étaient incomplètes pour **21 106 observations**.

Les données manquantes ont été complétées à partir de l’API historique **Open-Meteo**, en utilisant :

- latitude ;
- longitude ;
- date.

Les principales variables récupérées sont :

- température maximale ;
- précipitations ;
- vitesse maximale du vent ;
- humidité relative moyenne.

L’humidité moyenne est calculée à partir des **24 valeurs horaires UTC** de la journée.

Les valeurs déjà présentes dans le fichier initial ont été conservées.

Une vérification sur **24 observations complètes**, réparties entre 2018 et 2025, a permis de contrôler la concordance des données météorologiques.

Les URL et réponses utilisées lors de la récupération ont été conservées afin d’assurer la traçabilité des données.

---

## 2. Nettoyage et contrôle

Le nettoyage a principalement été effectué avec :

- **Pandas**
- **NumPy**
- **Shapely**
- **PyProj**

Les contrôles réalisés concernent :

- les types de données ;
- les dates ;
- les valeurs manquantes ;
- les doublons ;
- la cohérence des mesures ;
- la cohérence géographique.

Aucun doublon exact n’a été identifié.

Aucune valeur manquante n’a été remplacée arbitrairement par une moyenne.

### Contrôle géographique

**Shapely** a été utilisé pour vérifier les coordonnées par rapport au contour géographique de référence du Maroc.

**PyProj** a été utilisé pour calculer les distances par rapport aux centres des grilles météorologiques.

Résultats :

- **22 788 observations conservées**
- **22 219 observations** situées dans le contour de référence
- **569 observations** signalées en bordure
- Ces 569 observations représentent **34 coordonnées distinctes**

Les coordonnées étant arrondies à **0,1°**, les points situés en bordure ont été conservés pour vérification.

---

## 3. Association et contrôle de la fusion

Les observations sont associées aux données météorologiques à partir de :

```text
latitude
longitude
date
```

La table enrichie est séparée en deux tables :

1. observations ;
2. données météorologiques.

Une nouvelle fusion est ensuite effectuée afin de contrôler la cohérence.

La jointure est validée en :

```text
one_to_one
```

Résultats :

- **100 % de couverture**
- aucune perte de ligne ;
- aucune duplication de ligne ;
- aucune modification des valeurs sources.

---

## 4. Comparaison des traitements des valeurs manquantes

Le CSV source conservé est complet : **aucune valeur manquante réelle** n'est imputée.
Pour comparer les méthodes, un échantillon de **1 200 lignes du train** est utilisé sur des copies,
avec environ **5 % de cellules météo masquées artificiellement par variable**.

Les méthodes comparées sont :

- remplacement par une constante ;
- moyenne ;
- médiane ;
- mode ;
- KNN avec **3 voisins**, après standardisation ;
- régression itérative avec BayesianRidge, après standardisation.

Les donneurs représentent 80 % de cet échantillon et les 20 % restants servent au contrôle interne.
Les imputateurs et les scalers sont ajustés uniquement sur les donneurs masqués.
Les MAE et RMSE sont calculées dans les unités d'origine, sur **12 cellules réservées par variable**.
Ce petit effectif sert à une comparaison exploratoire, sans désigner une méthode universellement meilleure.

La suppression de lignes, la suppression de colonnes très incomplètes, le mode catégoriel
et la catégorie `Inconnu` sont également examinés sur des copies.
Les mécanismes **MCAR, MAR et MNAR** sont expliqués ; la simulation ne permet pas de conclure
sur le mécanisme réel de collecte. La cible `feu` n'est jamais imputée.

## 5. Détection des valeurs atypiques et transformations

Les diagnostics sont calculés sur l'entraînement :

- Z-score, avec un signal au-delà de `|z| > 3` et une zone de vigilance entre 2 et 3 ;
- bornes IQR à `Q1 − 1,5 × IQR` et `Q3 + 1,5 × IQR` ;
- boxplots, histogrammes et nuage température/humidité ;
- Isolation Forest, LOF et distance KNN sur un échantillon de **3 000 lignes du train**.

Les effets d'une suppression hypothétique, du capping, de `log1p` et de la racine carrée
sont comparés sur des copies. **Toutes les observations météo réelles physiquement plausibles sont conservées.**

Pour les précipitations du train, **Q1 = Q3 = 0** : la règle IQR signale les **3 825 pluies positives**.
Un capping automatique à cette borne effacerait toute la pluie. Ce traitement n'est donc pas appliqué
au dataset réel. Un signal statistique ne prouve pas une erreur de mesure.

## 6. Mise à l'échelle et encodage

Quatre méthodes sont comparées séparément :

| Méthode | Paramètres appris sur le train |
| --- | --- |
| MinMaxScaler | Minimum et maximum |
| MaxAbsScaler | Maximum de la valeur absolue |
| RobustScaler | Médiane et intervalle interquartile |
| StandardScaler | Moyenne et écart-type de population, `ddof=0` |

Leurs statistiques sont enregistrées pour examiner les effets sur les échelles.
Les mêmes paramètres sont réutilisés pour la validation et le test.

Les encodages examinés sont :

- **OneHotEncoder** pour la saison, sans ordre numérique entre les catégories ;
- **LabelEncoder** sur une copie textuelle de la cible ; la cible réelle reste codée 0/1 ;
- **OrdinalEncoder** avec un ordre explicite sur des classes exploratoires d'humidité.

Les seuils d'humidité utilisés dans cette comparaison ne définissent pas des niveaux validés de risque.
Le pipeline final conserve l'humidité continue. Il utilise trois colonnes de saison,
avec une quatrième saison comme référence.

## 7. Répartition des données et validation

La séparation principale respecte la chronologie, avant toute transformation apprise :

| Ensemble | Période | Observations | `feu=0` | `feu=1` |
| --- | --- | ---: | ---: | ---: |
| Entraînement | 2018–2023 | 17 904 | 11 582 | 6 322 |
| Validation | 2024 | 2 374 | 1 550 | 824 |
| Test | 2025 | 2 510 | 1 629 | 881 |

Des partitions reproductibles avec `random_state=42` et `stratify=y`, ainsi qu'un k-fold stratifié,
sont examinés **uniquement à l'intérieur du train**. Ils ne remplacent pas le protocole temporel principal.
Deux fenêtres chronologiques internes évaluent aussi 2022 puis 2023 après apprentissage sur les années précédentes.

## 8. Sélection exploratoire et réduction de dimension

Une analyse **SelectKBest** examine les scores F univariés et illustre la sélection de six variables.
Une **PCA** ajustée sur le train conserve **sept composantes**, représentant environ **95,78 % de la variance**.
Ces analyses sont exportées pour comparaison ; elles ne sont pas utilisées dans le pipeline final,
qui conserve ses **11 variables transformées**.

---

# 📁 Structure du projet

```text
morocco-wildfire-prediction/
├── Analyse_et_Modelisation/
│   ├── data/
│   ├── resultats/
│   ├── 01_Collecte_Nettoyage_Fusion.ipynb
│   ├── 02_Feature_Engineering_et_Split.ipynb
│   ├── 03_Modelisation.ipynb
│   ├── 04_Preparation_Donnees.ipynb
│   ├── preparer_donnees.py
│   ├── tester_prediction.py.ipynb
│   ├── requirements.txt
│   └── README.md
├── Preparation_Donnees/
│   ├── resultats/
│   ├── pipeline_pretraitement.py
│   ├── comparaisons.py
│   ├── aberrants.py
│   ├── construire_notebook.py
│   ├── executer_notebook.py
│   ├── verifier_resultats.py
│   ├── GUIDE_PRETRAITEMENT.md
│   └── README.md
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

[Analyse_et_Modelisation](Analyse_et_Modelisation/) regroupe les quatre notebooks principaux,
les sources et les premiers résultats.

[Preparation_Donnees](Preparation_Donnees/README.md) contient les scripts de prétraitement,
le guide des choix méthodologiques et le pipeline complet reproductible.

Le [guide de prétraitement](Preparation_Donnees/GUIDE_PRETRAITEMENT.md) détaille les méthodes,
les comparaisons et les limites des données.

---

# 🧪 Feature Engineering

Le pipeline construit effectivement :

- la **saison** à partir de la date ;
- **`jour_sin` et `jour_cos`**, pour représenter le cycle annuel et la proximité entre décembre et janvier ;
- **`log1p(precipitation_sum)`** dans les configurations comparant une transformation de la pluie.

Les coordonnées géographiques et les quatre variables météo constituent les autres entrées.
Le cycle annuel tient compte des années bissextiles.

Le pipeline reçoit **sept colonnes brutes** : `lat_c`, `lon_c`, `acq_date`,
`temperature_2m_max`, `precipitation_sum`, `wind_speed_10m_max` et `humidite_moy`.
Il produit huit colonnes numériques et trois indicatrices de saison.
Les identifiants, la cible et les métadonnées d'audit sont exclus des prédicteurs.
Aucune interaction supplémentaire ni variable météo glissante n'est inventée.

---

# 🤖 Modélisation

## Comparaison des prétraitements et modèle retenu

Le [notebook de préparation](Analyse_et_Modelisation/04_Preparation_Donnees.ipynb)
regroupe les contrôles, les comparaisons et une régression logistique pondérée par les classes.

**Sept configurations** sont comparées sur 2024 : aucune mise à l'échelle, Standard,
MinMax, MaxAbs et Robust sur la pluie brute, puis Standard et Robust avec `log1p(pluie)`.
Les paramètres de chaque pipeline sont ajustés sur **2018–2023 uniquement**.

Le choix repose sur l'**average precision de validation**, puis sur la ROC-AUC en cas d'égalité.
La configuration retenue est **`standard_log1p_pluie`** : logarithme de la pluie,
imputation numérique par médiane, standardisation, encodage de la saison et régression logistique.
Sur la validation 2024, son average precision vaut **0,4033** et sa ROC-AUC **0,5880**.

Le choix est figé avant l'évaluation de 2025, avec un **seuil de classification fixé à 0,5**
et sans réentraînement sur la validation.

## Résultats du test final 2025

| Métrique | Pipeline retenu | Référence : fréquence du train |
| --- | ---: | ---: |
| ROC-AUC | 0,5861 | 0,5000 |
| Average precision | 0,4117 | 0,3510 |
| Balanced accuracy | 0,5640 | 0,5000 |
| Précision | 0,4039 | 0,0000 |
| Rappel | 0,6345 | 0,0000 |
| F1 | 0,4936 | 0,0000 |

La référence prédit la fréquence observée dans le train ; au seuil 0,5, elle classe toutes les observations en 0.
Sa précision est reportée à zéro par convention lorsqu'aucune observation n'est prédite positive.

Ces résultats restent modestes : ils ne valident pas un dispositif opérationnel de prévision.
Les mesures météo couvrent la journée de la cible et ne sont pas toutes connues au début de cette journée.

Sources des résultats : [comparaison sur validation](Preparation_Donnees/resultats/comparaison_validation.csv)
et [évaluation finale](Preparation_Donnees/resultats/evaluation_test_final.csv).

## Premières comparaisons de modèles

Les notebooks [02](Analyse_et_Modelisation/02_Feature_Engineering_et_Split.ipynb)
et [03](Analyse_et_Modelisation/03_Modelisation.ipynb) contiennent les premières expériences avec
**Random Forest, SVM calibré et XGBoost**, les rapports de classification,
les matrices de confusion et l'importance des variables.
XGBoost a aussi fait l'objet d'une recherche **GridSearchCV de 27 configurations sur trois plis**,
avec le F1 comme critère.

Ces expériences utilisent un autre découpage : **2018–2021 pour apprendre, 2022–2025 pour tester**.
Leurs scores ne sont donc pas directement comparables à ceux du pipeline évalué uniquement sur 2025.
Leur validation croisée initiale n'impose pas la chronologie à l'intérieur des plis.
L'ancien scaler sauvegardé après standardisation doit également être distingué du pipeline réutilisable.
Ces précautions sont détaillées dans le [README du prétraitement](Preparation_Donnees/README.md).

## Fichiers produits et réutilisation

| Sortie dans `Preparation_Donnees/resultats/` | Contenu |
| --- | --- |
| `pipeline_complet.joblib` | Caractéristiques, imputation, scaler, encodage et modèle dans un seul objet |
| `X_train_pretraite.csv`, `X_validation_pretraite.csv`, `X_test_pretraite.csv` | Matrices transformées avec identifiant d'alignement |
| `y_train.csv`, `y_validation.csv`, `y_test.csv` | Cibles associées aux mêmes observations |
| `index_repartition.csv` | Dates, coordonnées, identifiants et ensemble de chaque ligne |
| `repartition_chronologique.csv` | Effectifs, périodes et classes du train, de la validation et du test |
| `validation_temporelle_interne.csv` | Évaluations des fenêtres chronologiques internes 2022 et 2023 |
| `predictions_test.csv` | Cible observée, score du modèle et classe prédite sur 2025 |
| `comparaison_*.csv`, `*aberrants*.csv`, `statistiques_echelles_train.csv` | Comparaisons des méthodes et diagnostics |
| `extension_selection_train.csv`, `extension_pca_train.csv` | Analyses exploratoires de sélection et de dimension |
| `*.png` | Graphiques des valeurs atypiques, transformations et matrice de confusion |
| `rapport_application.json` | Contrôles, configuration retenue, métriques, versions et SHA-256 de la source |
| `verification_resultats.json` | Vérification indépendante des exports et du pipeline rechargé |

`id_observation` sert à l'alignement des CSV et ne doit pas être utilisé comme prédicteur.
Le pipeline sauvegardé reçoit les sept colonnes brutes dans leurs unités initiales.
Le module `pipeline_pretraitement.py` doit rester accessible à Python au rechargement.
Le [guide de prétraitement](Preparation_Donnees/GUIDE_PRETRAITEMENT.md) détaille cette réutilisation.

---

# 🔬 Reproductibilité

Le notebook de préparation contient **12 cellules de calcul non vides exécutées**, avec tableaux et trois graphiques enregistrés.
Le prétraitement se reproduit à partir du CSV figé, sans nouvelle requête météo.
Les scripts conservent les versions des bibliothèques et l'empreinte SHA-256 de la source.

La vérification indépendante contrôle :

- la conservation de la source ;
- l'alignement des matrices, cibles et identifiants sur les 22 788 observations ;
- l'identité entre les matrices exportées et les transformations du pipeline ;
- l'identité des prédictions après sauvegarde et rechargement sur des entrées brutes ;
- la prise en charge de colonnes réordonnées et d'une valeur météo absente.

L'accès au jour de l'année utilise `day_of_year`, compatible avec les versions pandas utilisées.
Les archives de récupération météo décrites ci-dessous sont remises séparément ; elles ne sont pas nécessaires
pour réexécuter le prétraitement avec les sources conservées dans ce dépôt.

Le dossier justificatif contient notamment :

- les données initiales ;
- les données complétées ;
- les scripts ;
- les notebooks ;
- les réponses API archivées ;
- les contrôles de concordance ;
- les résultats de nettoyage ;
- les résultats des jointures ;
- les journaux d’exécution ;
- un manifeste **SHA-256**.

---

# 📦 Dossier justificatif

Une archive complémentaire est remise séparément :

```text
Dossier_Justificatif_EMSI_3_Etapes.zip
```

Elle contient notamment :

### Données

- CSV initial incomplet ;
- CSV complété ;
- dataset préparé ;
- informations de provenance.

### Code

- scripts Python ;
- notebooks Jupyter ;
- scripts de préparation.

### Preuves de récupération

- **6 212 réponses API mises en cache**
- contrôles de concordance ;
- état final de récupération.

### Contrôles

- résultats du nettoyage ;
- tables observations/météo ;
- points géographiques en bordure ;
- rapports de fusion.

### Reproductibilité

- journaux d’exécution ;
- contrôle de conservation des valeurs ;
- manifeste SHA-256.

---

# 🛠️ Technologies utilisées

Le projet utilise principalement :

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Shapely
- PyProj
- Open-Meteo API

---

# ⚙️ Installation

Cloner le dépôt puis entrer dans le dossier :

```bash
git clone https://github.com/anouar-baoutoul/morocco-wildfire-prediction.git
cd morocco-wildfire-prediction
```

Installer l'environnement déclaré par `pyproject.toml` et `uv.lock` :

```bash
uv sync --locked
```

Le projet utilise Python 3.13 ou une version compatible avec la contrainte déclarée.

---

# 🚀 Utilisation

Commencer par [04_Preparation_Donnees.ipynb](Analyse_et_Modelisation/04_Preparation_Donnees.ipynb),
sélectionner l'environnement `.venv` du projet et exécuter les cellules dans l'ordre.
Ce notebook est autonome et ses sorties sont déjà enregistrées.

Depuis la racine du dépôt, le même traitement se lance en script :

```bash
uv run python Preparation_Donnees/pipeline_pretraitement.py
uv run python Preparation_Donnees/verifier_resultats.py
```

Les notebooks `01`, `02` et `03` documentent les premiers traitements de collecte,
construction des variables et comparaison des modèles. Leur documentation est disponible
dans [Analyse_et_Modelisation/README.md](Analyse_et_Modelisation/README.md).

Les résultats du workflow complet sont enregistrés dans `Preparation_Donnees/resultats/`.

---

# ⚠️ Limites actuelles

Plusieurs améliorations restent possibles.

### Provenance des détections

Il serait utile de documenter précisément :

- le satellite ou capteur utilisé ;
- les filtres de qualité ;
- la méthode de construction des exemples `feu = 0`.

### Occupation du sol

Les observations pourraient être croisées avec des données d’occupation du sol afin de distinguer :

- forêts ;
- végétation ;
- zones agricoles ;
- zones urbaines.

### Horizon de prédiction

Certaines variables météorologiques, comme la température maximale de la journée, ne sont pas nécessairement disponibles au début de cette journée.

Une future version devra donc définir clairement l’**horizon de prévision**.

### Séries temporelles

Il serait intéressant de créer des variables comme :

- cumul de pluie sur 7 jours ;
- humidité des jours précédents ;
- température moyenne récente ;
- nombre de jours sans pluie ;
- vent moyen ou maximal sur plusieurs jours.

### Validation

La séparation **temporelle est déjà appliquée** dans le pipeline : apprentissage jusqu'en 2023,
validation en 2024 et test en 2025.
Une validation **spatiale** sur des lieux absents de l'entraînement reste à construire,
car certains lieux peuvent être présents dans plusieurs périodes.
Les probabilités produites ne sont pas calibrées sur la prévalence réelle des incendies dans la population.

### Surface brûlée

La régression de la surface brûlée nécessitera l’ajout d’une variable cible adaptée.

---

# 🌍 Sources

### NASA FIRMS

NASA FIRMS fournit des données satellitaires relatives aux feux actifs.

- [NASA FIRMS – données par pays](https://firms.modaps.eosdis.nasa.gov/country/)
- [NASA FIRMS – description des données VIIRS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)

### Open-Meteo

Les données météorologiques historiques utilisées dans le projet proviennent de :

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo – Terms](https://open-meteo.com/en/terms)

### geoBoundaries

Le contrôle géographique utilise un contour de référence du Maroc :

- [geoBoundaries – MAR ADM0](https://www.geoboundaries.org/api/current/gbOpen/MAR/ADM0/)

Version utilisée :

```text
9469f09
```

Identifiant :

```text
MAR-ADM0-5654995
```

Source indiquée :

```text
OpenStreetMap / Wambacher
```

Licence :

[OpenStreetMap / ODbL](https://www.openstreetmap.org/copyright)

---

# 👨‍💻 Auteurs

**Adam TALIBI**  
**Anouar BAOUTOUL**

**EMSI — 2026/2027**

---

# ✅ État du projet

- ✅ Collecte des données
- ✅ Complément des données météorologiques
- ✅ Nettoyage
- ✅ Contrôle géographique
- ✅ Fusion observations/météo
- ✅ Préparation du dataset
- ✅ Feature engineering
- ✅ Séparation des données
- ✅ Comparaison des imputations, valeurs atypiques et transformations
- ✅ Encodages et comparaison des quatre scalers
- ✅ Séparation train / validation / test chronologique
- ✅ Comparaison de sept configurations sur validation
- ✅ Évaluation finale sur 2025 et référence simple
- ✅ Pipeline complet sauvegardé et vérifié
- ✅ Analyses exploratoires de sélection et PCA
- 🔄 Amélioration des performances et validation spatiale

---

⭐ **Morocco Wildfire Prediction — EMSI 2026/2027**
