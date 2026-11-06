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
- **17 colonnes**
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

Après la préparation des données, différentes variables peuvent être construites afin d’améliorer la prédiction.

Exemples :

- variables temporelles ;
- mois ;
- saison ;
- caractéristiques météorologiques ;
- informations géographiques ;
- interactions entre certaines variables.

Une attention particulière doit être portée au risque de **data leakage**.

---

# 🤖 Modélisation

Le [notebook de préparation](Analyse_et_Modelisation/04_Preparation_Donnees.ipynb)
contient un workflow complet : contrôle des données, comparaisons des imputations,
valeurs atypiques, mise à l'échelle, encodage, entraînement et évaluation.

Le protocole utilise **2018–2023 pour l'entraînement, 2024 pour la validation et 2025 pour le test**.
Les transformations sont ajustées sur l'entraînement uniquement. Une régression logistique
sert à comparer les prétraitements sur la validation, avant l'évaluation finale sur le test.

Le fichier `Preparation_Donnees/resultats/pipeline_complet.joblib` réunit les transformations
et le modèle retenus pour recevoir directement les entrées brutes.

Les premières comparaisons de modèles restent disponibles dans
[03_Modelisation.ipynb](Analyse_et_Modelisation/03_Modelisation.ipynb).
Le [README du prétraitement](Preparation_Donnees/README.md) explique les précautions
concernant la réutilisation de ses anciens fichiers de modèle et de scaler.

Les métriques comprennent précision, rappel, F1, balanced accuracy, ROC-AUC,
average precision et matrice de confusion.

---

# 🔬 Reproductibilité

La préparation des données a été réexécutée.

La récupération météorologique peut être rejouée à partir des données mises en cache sans effectuer de nouvelles requêtes.

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

Pour obtenir une évaluation plus réaliste, une séparation :

- **temporelle**
- et éventuellement **spatiale**

est recommandée.

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
- ✅ Première étape de modélisation
- 🔄 Évaluation et amélioration des modèles

---

⭐ **Morocco Wildfire Prediction — EMSI 2026/2027**
