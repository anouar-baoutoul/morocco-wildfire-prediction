# Prédiction des feux de forêt au Maroc

Projet réalisé par **Adam TALIBI et Anouar BAOUTOUL**, à l’**EMSI**, dans le cadre de l’année universitaire **2026/2027**.

## Présentation du projet

L’objectif du projet est d’étudier des observations associées aux feux de forêt au Maroc ainsi que les conditions météorologiques correspondantes afin de préparer un modèle de **classification permettant de prédire la présence ou l’absence d’un feu**.

L’estimation de la **surface brûlée** constitue également une piste d’évolution du projet. Cependant, cette partie nécessiterait une variable cible supplémentaire qui n’est pas disponible dans la base de données actuelle.

---

## Données utilisées

La base de données préparée contient :

- **22 788 observations**
- Une période allant de **2018 à 2025**
- **1 232 coordonnées géographiques distinctes**
- **17 colonnes**

Les 17 colonnes correspondent aux **13 variables initiales** ainsi qu’à quatre informations supplémentaires permettant d’assurer le suivi et la traçabilité des données :

- identifiant ;
- origine des données météorologiques ;
- statut géographique ;
- distance par rapport à la grille météorologique.

### Variable cible

La variable cible utilisée pour la classification est :

```text
feu
```

Répartition des observations :

- `feu = 0` : **14 761 observations**
- `feu = 1` : **8 027 observations**

Ces proportions représentent uniquement la distribution de **notre échantillon de données**. Elles ne doivent pas être interprétées comme la probabilité réelle d’apparition d’un incendie au Maroc.

---

# Les trois premières étapes

## 1. Collecte et complément des données météorologiques

Le projet démarre à partir du fichier CSV fourni.

La provenance des détections présentes dans ce fichier est déclarée comme provenant de **NASA FIRMS**.

Cependant, l’export satellitaire brut d’origine n’étant pas disponible, nous ne présentons pas de nouveau téléchargement FIRMS comme ayant été effectué.

### Complément des données météo

Les informations météorologiques étaient incomplètes pour :

**21 106 observations**

Les données historiques manquantes ont été récupérées depuis l’API **Open-Meteo Historical Weather** à partir de :

- la latitude ;
- la longitude ;
- la date.

Les variables météorologiques récupérées sont notamment :

- température maximale ;
- précipitations ;
- vitesse maximale du vent ;
- humidité relative moyenne.

L’humidité relative moyenne est calculée à partir des **24 valeurs horaires UTC de la journée**.

Les valeurs météorologiques déjà présentes dans le fichier d’origine ont été conservées.

### Contrôle de concordance

Une vérification a été réalisée sur **24 observations initialement complètes**, réparties entre **2018 et 2025**.

Cette vérification a permis de confirmer la concordance des **six valeurs météorologiques et de grille comparées** avec la configuration utilisée.

Pour assurer la traçabilité :

- les URL des requêtes ont été conservées ;
- les réponses de récupération ont été archivées ;
- les données déjà existantes n’ont pas été remplacées inutilement.

---

## 2. Nettoyage et contrôle des données

Le nettoyage des données a principalement été réalisé avec :

- **Pandas**
- **NumPy**
- **Shapely**
- **PyProj**

### Contrôles effectués

Les vérifications ont porté sur :

- les types de données ;
- les dates ;
- les valeurs manquantes ;
- les doublons ;
- la cohérence des mesures ;
- la cohérence géographique.

Aucun doublon exact n’a été identifié.

Aucune valeur manquante n’a été remplacée arbitrairement par une moyenne.

### Vérification géographique

La bibliothèque **Shapely** a été utilisée pour vérifier la position des coordonnées par rapport au contour géographique de référence du Maroc.

**PyProj** a été utilisé pour calculer les distances entre les observations et les centres des grilles météorologiques.

### Résultats

Après préparation :

- **22 788 lignes** sont conservées ;
- aucune valeur manquante ne subsiste dans le fichier préparé ;
- **22 219 observations** ont leur centre situé dans le contour de référence ;
- **569 observations** sont signalées en bordure ;
- ces 569 observations correspondent à **34 coordonnées distinctes**.

Les observations situées en bordure ont été conservées afin de permettre une vérification ultérieure.

### Précision des coordonnées

Les coordonnées sont arrondies à :

**0,1°**

Le recoupement entre les cellules obtenues par cet arrondi et le contour géographique est compatible avec cette précision.

Cette vérification ne permet toutefois pas de certifier la position géographique exacte des détections satellitaires originales.

---

## 3. Association et contrôle de la fusion

Les données météorologiques récupérées ont été associées aux observations à partir de trois clés :

```text
latitude
longitude
date
```

Afin de contrôler correctement l’opération, la table enrichie est séparée en deux tables :

1. une table contenant les observations ;
2. une table contenant les informations météorologiques.

Les deux tables sont ensuite fusionnées de nouveau à l’aide des mêmes clés.

### Contrôles de la jointure

La jointure est contrôlée en relation :

**un-à-un**

Les résultats obtenus sont :

- couverture de la jointure : **100 %** ;
- aucune multiplication du nombre de lignes ;
- aucune perte d’observation ;
- aucune modification des valeurs sources.

Cette séparation puis cette nouvelle fusion constituent un **contrôle de cohérence de la table enrichie**.

Elle ne remplace cependant pas une validation indépendante utilisant directement l’export satellitaire brut original.

---

# Organisation du projet

La structure principale du dépôt est la suivante :

```text
morocco-wildfire-prediction/
│
├── Etapes 1+2+3/
│   ├── data/
│   ├── results/
│   ├── 01_Collecte_Nettoyage_Fusion.ipynb
│   ├── 02_Feature_Engineering_et_Split.ipynb
│   ├── 03_Modelisation.ipynb
│   ├── preparer_donnees.py
│   ├── tester_prediction.py
│   ├── README.md
│   └── requirements.txt
│
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

Les fichiers correspondant aux trois premières étapes sont accessibles dans :

[Etapes 1+2+3](Etapes%201%2B2%2B3/)

Un README plus détaillé est également disponible dans :

[README des étapes 1+2+3](Etapes%201%2B2%2B3/README.md)

---

# Dossier justificatif des trois étapes

Une archive complémentaire est remise séparément :

```text
Dossier_Justificatif_EMSI_3_Etapes.zip
```

Cette archive regroupe les éléments permettant de justifier les traitements réalisés.

> Tous les fichiers de cette archive ne sont pas nécessairement décompressés directement dans le dépôt Git.

Elle contient notamment les éléments suivants.

### Rapport et recherches

- synthèse des trois premières étapes ;
- description des limites ;
- liens vers les sources ;
- explication des choix techniques.

### Données

- CSV initial incomplet ;
- CSV complété ;
- fichier préparé ;
- informations détaillées sur la provenance des données météorologiques.

### Codes

- script de récupération ;
- script de préparation ;
- notebooks exécutés ;
- résultats obtenus.

### Preuves de récupération

L’archive contient notamment :

**6 212 réponses API mises en cache**

Elle comprend également :

- les contrôles de concordance ;
- l’état final de récupération ;
- les informations permettant de vérifier la provenance des données.

### Résultats de nettoyage et de fusion

- tables d’observations ;
- tables météorologiques ;
- points identifiés en bordure ;
- rapports de contrôle.

### Reproductibilité

Le dossier comprend également :

- les journaux d’exécution ;
- les vérifications de conservation des valeurs ;
- un manifeste **SHA-256** des fichiers.

### Carte

Une carte représente les :

**1 232 lieux étudiés**

Elle permet également d’identifier les points signalés en bordure du contour géographique.

---

# Reproductibilité

La préparation des données a été réexécutée.

La récupération météorologique a également été rejouée à partir du **cache local**, sans effectuer de nouvelles requêtes vers l’API.

Les données complétées obtenues sont identiques à celles présentes dans le dossier justificatif.

Cette démarche permet d’améliorer :

- la reproductibilité ;
- la traçabilité ;
- la vérification des traitements réalisés.

---

# Technologies utilisées

Le projet utilise principalement :

```text
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Shapely
PyProj
Jupyter Notebook
```

Les données météorologiques historiques sont obtenues grâce à :

**Open-Meteo Historical Weather API**

Le contrôle géographique utilise un contour du Maroc provenant de :

**geoBoundaries**

La provenance déclarée des observations d’incendies est :

**NASA FIRMS**

---

# Points à approfondir

Plusieurs améliorations pourront être apportées dans les prochaines étapes du projet.

## Provenance des observations

Documenter plus précisément :

- le capteur satellitaire utilisé ;
- les filtres de qualité appliqués ;
- la méthode utilisée pour construire les observations `feu = 0`.

## Vérification des zones forestières

Croiser les détections avec des données d’occupation du sol afin de vérifier que les observations correspondent réellement à des zones forestières ou végétalisées.

## Horizon de prévision

Définir précisément à quel moment la prédiction doit être réalisée.

En effet, certaines variables, comme la **température maximale de la journée**, ne sont pas connues au début de cette même journée.

Une véritable prédiction opérationnelle devra donc utiliser uniquement des variables disponibles au moment où la prévision est effectuée.

## Séries temporelles

Reconstituer de véritables séries météorologiques quotidiennes avant de calculer des indicateurs tels que :

- pluie cumulée sur 7 jours ;
- température moyenne des jours précédents ;
- vent moyen ;
- humidité moyenne ;
- nombre de jours sans pluie.

## Séparation des données

Pour éviter les fuites d’information entre les données d’entraînement et de test, il sera préférable d’utiliser :

- une séparation temporelle ;
- et, si possible, une évaluation spatiale.

## Régression de la surface brûlée

Ajouter une variable représentant la **surface brûlée** afin de permettre la réalisation de la partie régression prévue dans le sujet.

---

# Sources et attribution

## NASA FIRMS

NASA FIRMS permet d’accéder aux informations relatives aux détections satellitaires de feux actifs.

- [NASA FIRMS — données par pays](https://firms.modaps.eosdis.nasa.gov/country/)
- [Description des attributs VIIRS FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html)

## Open-Meteo

Les données météorologiques historiques utilisées pour compléter la base proviennent de l’API Open-Meteo.

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Conditions d’utilisation Open-Meteo](https://open-meteo.com/en/terms)

Les données sont attribuées à **Open-Meteo** conformément aux conditions applicables.

## geoBoundaries

Le contour géographique utilisé pour les vérifications correspond à la couche :

```text
MAR ADM0
```

Version utilisée :

```text
9469f09
```

Identifiant :

```text
MAR-ADM0-5654995
```

Source déclarée :

```text
OpenStreetMap / Wambacher
```

Référence :

- [geoBoundaries — Maroc ADM0](https://www.geoboundaries.org/api/current/gbOpen/MAR/ADM0/)
- [Licence OpenStreetMap / ODbL](https://www.openstreetmap.org/copyright)

Le contour est utilisé uniquement comme **référence technique pour le contrôle géographique**.

---

# Auteurs

**Adam TALIBI**  
**Anouar BAOUTOUL**

**EMSI — 2026/2027**

---

## État actuel du projet

Les étapes suivantes sont réalisées :

- ✅ collecte et complément des données ;
- ✅ nettoyage des données ;
- ✅ contrôle géographique ;
- ✅ association météo / observations ;
- ✅ contrôle de la fusion ;
- ✅ préparation du dataset ;
- ✅ feature engineering et séparation des données ;
- ✅ préparation de la modélisation.

La suite du projet porte principalement sur la **modélisation, l’évaluation et l’amélioration des performances de prédiction des feux de forêt**.
