# Prédiction des feux de forêt au Maroc

Projet réalisé par **Adam TALIBI et Anouar BAOUTOUL**, à l’EMSI, pour l’année 2026/2027.

Nous étudions des observations de feux et les conditions météorologiques associées pour préparer un modèle de classification. L’estimation de la surface brûlée reste un objectif possible, mais nécessite une cible supplémentaire absente de notre base actuelle.

## Données utilisées

La base préparée contient **22 788 observations**, couvrant **2018 à 2025**, pour **1 232 coordonnées distinctes**. Elle comporte 17 colonnes : les 13 colonnes initiales et quatre informations de suivi (identifiant, origine météo, statut géographique et distance à la grille météo).

La cible `feu` reprend les étiquettes du fichier fourni : 14 761 lignes codées `0` et 8 027 codées `1`. Ces proportions décrivent notre échantillon, pas le risque réel d’incendie au Maroc.

## Les trois premières étapes

### 1. Collecte et complément des données météo

Nous sommes partis du CSV fourni pour le projet, dont la provenance des détections est déclarée NASA FIRMS. L’export satellitaire brut n’étant pas disponible, aucun nouveau téléchargement FIRMS n’est présenté comme réalisé.

La météo était incomplète sur **21 106 lignes**. Nous avons récupéré les données historiques correspondantes auprès d’Open-Meteo, en utilisant la latitude, la longitude et la date : température maximale, précipitations, vent maximal et humidité relative moyenne. L’humidité est calculée à partir des 24 valeurs horaires UTC.

Les valeurs déjà présentes ont été conservées. Une comparaison sur 24 observations initialement complètes, réparties entre 2018 et 2025, a confirmé la concordance des six valeurs météo et de grille comparées avec la configuration testée. Les URL et les réponses de récupération ont été archivées pour conserver la provenance.

### 2. Nettoyage et contrôle

Avec **Pandas et NumPy**, nous avons vérifié les types, les dates, les valeurs manquantes, les doublons et la cohérence des mesures. Aucun doublon exact n’a été trouvé et aucune valeur n’a été remplacée arbitrairement par une moyenne.

**Shapely** a servi à vérifier les coordonnées par rapport au contour de référence. **PyProj** a permis de calculer les distances aux centres de grille météo.

Résultats :

- 22 788 lignes conservées et aucune valeur manquante dans le fichier préparé ;
- 22 219 observations dont le centre est dans le contour ;
- 569 observations, réparties sur 34 coordonnées, signalées en bordure et conservées pour vérification.

Les coordonnées sont arrondies à 0,1°. Le recoupement de leurs cellules d’arrondi avec le contour est compatible avec cette imprécision, mais ne certifie pas la position originale des détections.

### 3. Association et contrôle de la fusion

Les données météo récupérées ont été associées aux coordonnées et à la date de chaque ligne. La préparation sépare ensuite la table enrichie en deux tables — observations et météo — avant de vérifier leur jointure exacte sur latitude, longitude et date.

La jointure est contrôlée en **un-à-un**, avec une couverture de **100 %**, sans multiplication des lignes ni modification des valeurs sources. Cette séparation puis fusion est un contrôle de cohérence de la table enrichie ; elle ne remplace pas une validation à partir d’un export satellitaire brut.

Les fichiers de ces étapes sont accessibles dans [Etapes 1+2+3](Etapes%201%2B2%2B3/), avec un [README détaillé](Etapes%201%2B2%2B3/README.md).

## Contenu du dossier justificatif des trois étapes

L’archive remise séparément, `Dossier_Justificatif_EMSI_3_Etapes.zip`, regroupe les éléments suivants. Cette liste décrit l’archive complète ; tous ses fichiers ne sont pas nécessairement décompressés dans ce dépôt.

- **Rapport et recherches** : bilan des trois étapes, limites, liens des sources et explication des choix techniques.
- **Données** : CSV initial incomplet, CSV complété, fichier préparé et provenance détaillée de la récupération météo.
- **Codes** : script de récupération, script de préparation et notebook exécuté avec ses résultats.
- **Preuves de récupération** : 6 212 réponses API mises en cache, contrôles de concordance et état final de récupération.
- **Résultats de nettoyage et de fusion** : tables observations/météo, points en bordure et rapport de contrôles.
- **Reproductibilité** : journaux d’exécution, vérification de conservation des valeurs et manifeste SHA-256 des fichiers.
- **Carte** : représentation des 1 232 lieux étudiés, avec signalement des points en bordure.

La préparation a été réexécutée et la récupération a été rejouée depuis le cache, sans nouvelles requêtes : les données complétées obtenues étaient identiques à celles du dossier.

## Ajout : analyse univariée

Le [ZIP de l’analyse univariée](Analyse_univariee/Analyse_Univariee_EMSI.zip) contient :

- un **rapport PDF de 18 pages**, avec une introduction et une fiche pour chacune des 17 variables ;
- le script Python permettant de reproduire les calculs et le rapport ;
- les statistiques et tableaux de fréquences au format CSV ;
- le dataset préparé utilisé pour l’analyse et un guide de reproduction.

Le premier rendu porte en détail sur **la température maximale**, variable numérique, et **la présence de feu**, variable qualitative. L’analyse est ensuite étendue aux autres colonnes en distinguant mesures, catégories, dates et identifiants.

Pour chaque variable, nous présentons les mesures adaptées, leur interprétation, le choix d’une éventuelle normalisation et une visualisation. Les graphiques retenus sont les histogrammes, les boîtes à moustaches et les diagrammes en barres. Les dates sont décrites par leurs effectifs annuels.

La version actuelle du PDF utilise des textes retravaillés, des graphiques bleus sur fond blanc et des axes noirs. La ligne nominative répétée en pied de page a été supprimée. Ces modifications de présentation ne changent pas les données ni les calculs.

La normalisation est **proposée et expliquée**, sans modifier le dataset : centrage-réduction lorsque le modèle le nécessite, transformation logarithmique envisagée pour les précipitations, division par 100 pour l’humidité et encodage cyclique pour le calendrier. La cible `feu` et les identifiants ne sont pas normalisés. Les paramètres appris devront être ajustés uniquement sur le jeu d’entraînement.

## Reproduire les résultats

Pour la préparation, ouvrir un terminal dans `Etapes 1+2+3`, puis lancer :

```bash
python -m pip install -r requirements.txt
python preparer_donnees.py
```

Les données nécessaires à cette préparation sont dans le sous-dossier `data`. Le notebook `01_Collecte_Nettoyage_Fusion.ipynb` contient aussi les sorties enregistrées.

Pour l’analyse univariée, extraire son ZIP et ouvrir un terminal dans le dossier extrait :

```bash
python -m pip install pandas numpy reportlab
python analyse_univariee.py
```

Le dataset doit rester à côté du script. La version actuelle du générateur PDF utilise les polices Calibri et Georgia de Windows, dans `C:/Windows/Fonts` ; adapter leurs chemins pour un autre système.

Le dépôt contient également des notebooks de préparation des variables, de modélisation et de test de prédiction. Le présent bilan documente les trois premières étapes et le nouveau rendu univarié ; il ne constitue pas une validation des résultats de ces autres notebooks.

## Points à approfondir

- Documenter le capteur, les filtres de qualité et la construction des exemples `feu=0` à partir de la source initiale.
- Vérifier le caractère forestier des détections avec des données d’occupation du sol.
- Définir un horizon de prévision : les maxima météo d’une journée ne sont pas connus au début de cette journée.
- Reconstituer de vraies séries quotidiennes avant de calculer des indicateurs sur les sept jours précédents.
- Utiliser une séparation temporelle et, si possible, une évaluation spatiale.
- Ajouter une mesure de surface brûlée pour réaliser la régression prévue dans le sujet.

## Sources et attribution

- [NASA FIRMS : données par pays](https://firms.modaps.eosdis.nasa.gov/country/) et [description des attributs VIIRS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html).
- [Open-Meteo : archive météorologique historique](https://open-meteo.com/en/docs/historical-weather-api), données attribuées à Open-Meteo sous CC BY 4.0 ; consulter les [conditions d’utilisation](https://open-meteo.com/en/terms) pour toute nouvelle collecte.
- [geoBoundaries : métadonnées MAR ADM0](https://www.geoboundaries.org/api/current/gbOpen/MAR/ADM0/), contour figé à la version `9469f09`, identifiant `MAR-ADM0-5654995`, source OpenStreetMap/Wambacher. La couche utilisée indique la licence [ODbL 1.0](https://www.openstreetmap.org/copyright).

Le contour sert de référence technique au contrôle géographique.
