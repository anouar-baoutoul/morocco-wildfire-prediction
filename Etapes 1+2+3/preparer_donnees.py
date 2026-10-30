# %% Étape 1 : charger les sources conservées
from pathlib import Path
import json
import hashlib
import platform
import pandas as pd
import numpy as np
import shapely
import pyproj
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union

ROOT = Path.cwd()
DATA = ROOT / 'data'
OUT = ROOT / 'resultats'
OUT.mkdir(exist_ok=True)
source = DATA / 'dataset_feux_meteo_complete.csv'
raw = pd.read_csv(source, encoding='utf-8-sig')
original = raw.copy(deep=True)
required = ['lat_c','lon_c','acq_date','feu','mois','jour_annee','date',
            'temperature_2m_max','precipitation_sum','wind_speed_10m_max',
            'humidite_moy','lat_grille','lon_grille']
assert list(raw.columns) == required, 'Schéma inattendu : vérifier la source.'
raw.insert(0, 'id_observation', np.arange(1, len(raw)+1))
provenance = pd.read_csv(DATA/'provenance_meteo_retrouvee.csv')
recovered_ids = set(provenance['ligne_csv'].astype(int)-1)
raw['origine_meteo'] = np.where(raw.id_observation.isin(recovered_ids),
                               'recuperee_open_meteo','presente_dans_fichier_initial')
print(f'{len(raw):,} observations chargées ; {len(required)} colonnes sources.')
print('Source incendies déclarée : NASA FIRMS. Export satellitaire brut non fourni.')
print('Météo : archive Open-Meteo ; aucune nouvelle requête réseau dans ce notebook.')

# %% Étape 2 : types, doublons, dates et cohérence physique
numeric = [c for c in required if c not in ('date','acq_date')]
for c in numeric:
    raw[c] = pd.to_numeric(raw[c], errors='raise')
assert np.isfinite(raw[numeric].to_numpy()).all(), 'Valeur numérique absente ou infinie.'
for c in ('date','acq_date'):
    raw[c] = pd.to_datetime(raw[c], format='%Y-%m-%d', errors='raise')
assert not raw[required].isna().any().any()
assert raw.feu.isin([0,1]).all()
for c in ('feu','mois','jour_annee'):
    assert (raw[c] == raw[c].round()).all()
    raw[c] = raw[c].astype('int64')
assert raw.lat_c.between(-90,90).all() and raw.lon_c.between(-180,180).all()
assert raw.lat_grille.between(-90,90).all() and raw.lon_grille.between(-180,180).all()
assert raw.date.eq(raw.acq_date).all(), 'Dates incendie/météo non concordantes.'
assert raw.mois.eq(raw.acq_date.dt.month).all()
assert raw.jour_annee.eq(raw.acq_date.dt.dayofyear).all()
assert raw.humidite_moy.between(0,100).all()
assert raw.precipitation_sum.ge(0).all() and raw.wind_speed_10m_max.ge(0).all()
assert raw.temperature_2m_max.gt(-273.15).all()
# Les extrêmes plausibles sont conservés ; pas de suppression automatique par IQR.
duplicate_count = int(raw.duplicated(required).sum())
clean = raw.drop_duplicates(required, keep='first').copy()
key = ['lat_c','lon_c','acq_date']
assert not clean.duplicated(key).any(), 'Clés contradictoires : examen manuel nécessaire.'
print(f'Doublons exacts retirés : {duplicate_count}. Valeurs manquantes : 0.')
print(clean[['temperature_2m_max','precipitation_sum','wind_speed_10m_max','humidite_moy']].agg(['min','max']).to_string())

# %% Étape 2 : contrôle spatial tenant compte de la précision des coordonnées
geojson = json.loads((DATA/'maroc.geojson').read_text(encoding='utf-8'))
country = unary_union([shape(f['geometry']) for f in geojson['features']])
assert country.is_valid, 'Contour invalide : ne pas réparer sans examiner la géométrie.'
coords = clean[['lat_c','lon_c']].drop_duplicates().copy()
# Les coordonnées sont toutes quantifiées au dixième de degré.
assert np.allclose(coords.to_numpy()*10, np.round(coords.to_numpy()*10))
def geo_status(lat, lon):
    if country.covers(Point(lon, lat)):
        return 'centre_dans_contour'
    # Hypothèse de précision : ±0,05°. Ce recouvrement ne prouve pas l'origine du feu.
    if country.intersects(box(lon-.05,lat-.05,lon+.05,lat+.05)):
        return 'limite_a_verifier'
    return 'hors_emprise_a_verifier'
coords['statut_geo'] = [geo_status(lat,lon) for lat,lon in coords.itertuples(index=False,name=None)]
clean = clean.merge(coords, on=['lat_c','lon_c'], how='left', validate='many_to_one')
assert not clean.statut_geo.isna().any()
# Les cas ambigus sont conservés et isolés pour examen ; aucune relocalisation fictive.
border = clean[clean.statut_geo.ne('centre_dans_contour')].copy()
border.to_csv(OUT/'points_limites_a_verifier.csv', index=False, encoding='utf-8-sig',date_format='%Y-%m-%d')
print(clean.statut_geo.value_counts().to_string())
print('Cas limites conservés : demander les coordonnées satellitaires non arrondies pour confirmer.')

# %% Étape 3 : reconstruire les deux tables et vérifier une fusion exacte
# La source livrée est déjà fusionnée. Cette séparation permet de contrôler les clés,
# mais ne constitue pas une validation indépendante des fichiers satellitaires bruts.
obs_cols = ['id_observation','lat_c','lon_c','acq_date','feu','mois','jour_annee','statut_geo']
meteo_cols = ['lat_c','lon_c','date','temperature_2m_max','precipitation_sum',
              'wind_speed_10m_max','humidite_moy','lat_grille','lon_grille','origine_meteo']
observations = clean[obs_cols].copy()
meteo = clean[meteo_cols].copy()
assert not meteo.duplicated(['lat_c','lon_c','date']).any()
merged = observations.merge(meteo, how='left', left_on=key,
                            right_on=['lat_c','lon_c','date'], validate='one_to_one',indicator=True)
assert merged['_merge'].eq('both').all()
assert len(merged)==len(observations)
merged = merged.drop(columns='_merge').sort_values('id_observation').reset_index(drop=True)
geod = pyproj.Geod(ellps='WGS84')
_,_,dist = geod.inv(merged.lon_c.to_numpy(),merged.lat_c.to_numpy(),
                   merged.lon_grille.to_numpy(),merged.lat_grille.to_numpy())
merged['distance_grille_km'] = dist/1000
# La distance entre grille météo et point de requête est normale : conserver les deux.
merged = merged[['id_observation']+required+['origine_meteo','statut_geo','distance_grille_km']]
assert not merged.isna().any().any()
check = clean.sort_values('id_observation').reset_index(drop=True)
pd.testing.assert_frame_equal(check[required],merged[required],check_dtype=False)
observations.to_csv(OUT/'observations_nettoyees.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
meteo.to_csv(OUT/'meteo_nettoyee.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
merged.to_csv(OUT/'dataset_prepare.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
print(f'Fusion exacte : {len(merged):,}/{len(observations):,}, soit 100 %.')
print(f'Distance maximale point/grille météo : {merged.distance_grille_km.max():.2f} km.')

# %% Bilan reproductible et vérification des fichiers exportés
report = {
    'lignes_source':len(original),'lignes_preparees':len(merged),
    'doublons_exacts_supprimes':duplicate_count,'valeurs_manquantes':int(merged.isna().sum().sum()),
    'dates_min_max':[str(merged.acq_date.min().date()),str(merged.acq_date.max().date())],
    'classes':{str(k):int(v) for k,v in merged.feu.value_counts().items()},
    'controle_geographique':{str(k):int(v) for k,v in merged.statut_geo.value_counts().items()},
    'lignes_meteo_retrouvees':int(merged.origine_meteo.eq('recuperee_open_meteo').sum()),
    'couverture_fusion':1.0,'distance_grille_max_km':float(merged.distance_grille_km.max()),
    'sha256_source':hashlib.sha256(source.read_bytes()).hexdigest(),
    'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                'shapely':shapely.__version__,'pyproj':pyproj.__version__},
    'limites':['Export FIRMS/VIIRS brut et critères de qualité non fournis.',
               'Construction des exemples feu=0 non documentée.',
               'Points arrondis proches des limites à confirmer avec les coordonnées initiales.',
               'Absence de surface brûlée : régression de superficie non réalisable avec cette table seule.',
               'Météo du jour : ne pas utiliser comme information déjà connue au début de ce jour.']
}
(OUT/'rapport_controles.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
saved = pd.read_csv(OUT/'dataset_prepare.csv',encoding='utf-8-sig')
assert len(saved)==len(merged) and not saved.isna().any().any()
assert list(saved.columns)==list(merged.columns)
assert saved.id_observation.is_unique
for c in numeric:
    assert np.allclose(saved[c],merged[c],rtol=0,atol=1e-10)
print(json.dumps(report,ensure_ascii=False,indent=2))
