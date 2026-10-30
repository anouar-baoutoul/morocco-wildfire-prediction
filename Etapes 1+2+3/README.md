# Préparation du projet de prédiction des feux de forêt

Adam Talibi et Anouar Baoutoul — EMSI — Analyse des Données — 2026/2027

## Travail réalisé

1. Collecte et traçabilité : conservation du CSV complet, des liens de récupération météo
   et d'une version figée du contour géographique. Les détections FIRMS sont déclarées
   par la source fournie ; aucun nouveau téléchargement satellitaire n'a été effectué.
2. Nettoyage : types numériques et dates, valeurs absentes, doublons, clés contradictoires,
   calendrier, humidité, précipitations, vent et coordonnées. Les extrêmes plausibles
   sont conservés, sans écrêtage ni imputation arbitraire.
3. Fusion : séparation de la table livrée en observations et météo, puis jointure exacte
   sur latitude, longitude et date. Le contrôle exclut une multiplication des lignes et
   vérifie la conservation des valeurs. Cette opération contrôle une table déjà fusionnée ;
   elle ne remplace pas une validation à partir des exports satellitaires bruts.

## Résultats

- 22788 observations, du 2018-01-01 au 2025-12-31.
- Zéro doublon exact, zéro valeur manquante, fusion complète.
- Classes originales conservées : {'0': 14761, '1': 8027}.
- Contrôle géographique : {'centre_dans_contour': 22219, 'limite_a_verifier': 569}.
- La base préparée conserve les 13 variables sources et ajoute l'identifiant de ligne,
  l'origine de la météo, le statut géographique et la distance à la grille météo.
- Les variables d'audit et l'identifiant ne sont pas destinés à devenir automatiquement des prédicteurs.

Les centres proches des limites restent dans le dataset préparé avec un indicateur.
La cellule ±0,05 degré recoupe le contour ; cela est compatible avec l'arrondi observé,
sans prouver que l'observation satellitaire initiale se situait au Maroc.
Ne pas présenter ces centres comme une localisation exacte certifiée.

## Fichiers

- `01_Collecte_Nettoyage_Fusion.ipynb` : notebook exécuté, avec résultats enregistrés.
- `preparer_donnees.py` : même traitement sous forme de script.
- `data/` : sources figées et provenance, à conserver sans modification.
- `resultats/dataset_prepare.csv` : table principale après préparation.
- `resultats/observations_nettoyees.csv` et `meteo_nettoyee.csv` : tables utilisées pour la fusion.
- `resultats/points_limites_a_verifier.csv` : observations géographiques ambiguës.
- `resultats/rapport_controles.json` : bilan, versions et empreinte de la source.

## Reproduire

Extraire le ZIP dans un dossier. Ouvrir un terminal dans ce dossier, puis :

    python -m pip install -r requirements.txt
    python preparer_donnees.py

Pour le notebook, installer Jupyter séparément (`python -m pip install jupyterlab`),
puis l'ouvrir dans JupyterLab. Les sorties déjà présentes permettent de le consulter
sans réexécuter les calculs. Aucune clé API n'est nécessaire pour cette préparation.

## Points à résoudre avant l'entraînement

- Obtenir l'export FIRMS brut, le capteur, les filtres de confiance et la méthode de création des `feu=0`.
- Distinguer un point chaud satellitaire d'un feu de forêt confirmé ; une occupation du sol est nécessaire
  pour vérifier le caractère forestier. L'absence de détection ne garantit pas une absence réelle de feu.
- Définir l'horizon de prévision. La météo observée sur une journée complète ne peut pas être considérée
  connue au début de cette journée. Ne pas inventer des variables J-1 par simple décalage des lignes.
- Pour les indicateurs de sept jours, compléter une vraie série quotidienne, car la table actuelle
  est un échantillon de couples zone/date et non une série quotidienne continue par zone.
- Séparer les années avant les transformations apprises. Proposition : 2018–2023 / 2024 / 2025.
- Ajouter une cible de surface brûlée si la régression fait partie du livrable final.

## Sources et licences

- NASA FIRMS : https://firms.modaps.eosdis.nasa.gov/country/
- Attributs VIIRS : https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html
- Open-Meteo : https://open-meteo.com/en/docs/historical-weather-api
  Données récupérées sous CC BY 4.0. Humidité calculée comme moyenne des 24 valeurs horaires UTC.
  La récupération reproduit les valeurs originales sur 24 observations de contrôle, de 2018 à 2025.
- geoBoundaries, MAR-ADM0-5654995 : https://www.geoboundaries.org/api/current/gbOpen/MAR/ADM0/
  Version figée 9469f09 ; source OpenStreetMap/Wambacher, limites représentées en 2017.
  La licence indiquée dans les métadonnées de cette couche est ODbL 1.0 :
  https://www.openstreetmap.org/copyright . Conserver ces métadonnées et cette attribution.
  Le contour est un référentiel technique pour ce contrôle, sans prise de position sur les frontières.
