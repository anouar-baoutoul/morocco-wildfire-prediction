"""Préparation reproductible du jeu de données incendies.

Exécuter : python Preparation_Donnees/pipeline_pretraitement.py
Les paramètres appris proviennent de 2018–2023. Le test 2025 est évalué
uniquement après le choix sur la validation 2024. Aucun accès réseau.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, MaxAbsScaler, MinMaxScaler, OneHotEncoder,
    RobustScaler, StandardScaler,
)

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / "Analyse_et_Modelisation" / "data" / "dataset_feux_meteo_complete.csv"
OUT = BASE / "resultats"
METEO = ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max", "humidite_moy"]
ENTREES = ["lat_c", "lon_c", "acq_date"] + METEO
NUMERIQUES = ["lat_c", "lon_c"] + METEO + ["jour_sin", "jour_cos"]
SCALERS = {
    "sans_mise_echelle": None,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
}


def exporter(table: pd.DataFrame, nom: str, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{nom}.csv", index=False, encoding="utf-8-sig")


def charger_nettoyer(source: Path = SOURCE) -> tuple[pd.DataFrame, dict]:
    """Contrôles déterministes, sans estimation sur des données futures.

    Les valeurs météo absentes restent NaN pour l'imputation du pipeline.
    Les erreurs de schéma, de cible, de date ou de domaine arrêtent le calcul.
    Une valeur statistiquement rare et physiquement possible reste conservée.
    """
    brut = pd.read_csv(source, encoding="utf-8-sig")
    attendues = ["lat_c", "lon_c", "acq_date", "feu", "mois", "jour_annee", "date"] + METEO + ["lat_grille", "lon_grille"]
    if list(brut.columns) != attendues:
        raise ValueError("Schéma source inattendu. Vérifier les colonnes avant de poursuivre.")
    donnees = brut.copy(deep=True)
    donnees.insert(0, "id_observation", np.arange(1, len(donnees) + 1))
    for col in [c for c in attendues if c not in ("date", "acq_date")]:
        donnees[col] = pd.to_numeric(donnees[col], errors="raise")
        if np.isinf(donnees[col].to_numpy()).any():
            raise ValueError(f"Valeur infinie : {col}")
    for col in ("date", "acq_date"):
        donnees[col] = pd.to_datetime(donnees[col], format="%Y-%m-%d", errors="raise")
    indispensables = [c for c in attendues if c not in METEO]
    if donnees[indispensables].isna().any().any():
        raise ValueError("Une cible, date, coordonnée ou variable calendrier est absente.")
    controles = {
        "cible_binaire": donnees.feu.isin([0, 1]).all(),
        "dates_identiques": donnees.date.eq(donnees.acq_date).all(),
        "mois_coherent": donnees.mois.eq(donnees.acq_date.dt.month).all(),
        "jour_coherent": donnees.jour_annee.eq(donnees.acq_date.dt.day_of_year).all(),
        "latitude_valide": donnees.lat_c.between(-90, 90).all(),
        "longitude_valide": donnees.lon_c.between(-180, 180).all(),
        "latitude_grille_valide": donnees.lat_grille.between(-90, 90).all(),
        "longitude_grille_valide": donnees.lon_grille.between(-180, 180).all(),
        "humidite_valide": (donnees.humidite_moy.isna() | donnees.humidite_moy.between(0, 100)).all(),
        "pluie_valide": (donnees.precipitation_sum.isna() | donnees.precipitation_sum.ge(0)).all(),
        "vent_valide": (donnees.wind_speed_10m_max.isna() | donnees.wind_speed_10m_max.ge(0)).all(),
        "temperature_valide": (donnees.temperature_2m_max.isna() | donnees.temperature_2m_max.gt(-273.15)).all(),
    }
    erreurs = [k for k, v in controles.items() if not v]
    if erreurs:
        raise ValueError(f"Incohérences à examiner, aucune correction inventée : {erreurs}")
    doublons = int(donnees.duplicated(attendues).sum())
    donnees = donnees.drop_duplicates(attendues).copy()
    if donnees.duplicated(["lat_c", "lon_c", "acq_date"]).any():
        raise ValueError("Plusieurs observations contradictoires pour le même lieu et jour.")
    donnees["feu"] = donnees.feu.astype(int)
    donnees = donnees.sort_values(["acq_date", "id_observation"]).reset_index(drop=True)
    rapport = {
        "source": str(source), "sha256_source": hashlib.sha256(source.read_bytes()).hexdigest(),
        "lignes_source": len(brut), "lignes_nettoyees": len(donnees),
        "doublons_exacts_retires": doublons,
        "manquants_reels": {k: int(v) for k, v in donnees.isna().sum().items()},
        "controles": {k: bool(v) for k, v in controles.items()},
        "valeurs_extremes_reelles_supprimees": 0,
    }
    return donnees, rapport


def repartir(donnees: pd.DataFrame) -> dict[str, pd.DataFrame]:
    annees = donnees.acq_date.dt.year
    if not annees.between(2018, 2025).all():
        raise ValueError("Années inattendues : redéfinir explicitement les périodes.")
    parties = {
        "train": donnees.loc[annees.le(2023)].copy(),
        "validation": donnees.loc[annees.eq(2024)].copy(),
        "test": donnees.loc[annees.eq(2025)].copy(),
    }
    for nom, partie in parties.items():
        if partie.empty or partie.feu.nunique() != 2:
            raise ValueError(f"Ensemble {nom} vide ou sans les deux classes.")
    assert sum(len(p) for p in parties.values()) == len(donnees)
    assert parties["train"].acq_date.max() < parties["validation"].acq_date.min()
    assert parties["validation"].acq_date.max() < parties["test"].acq_date.min()
    ids = [set(p.id_observation) for p in parties.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    return parties


def bilan_repartition(parties: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"ensemble": nom, "lignes": len(p), "debut": str(p.acq_date.min().date()),
         "fin": str(p.acq_date.max().date()), "feu_0": int(p.feu.eq(0).sum()),
         "feu_1": int(p.feu.eq(1).sum()), "proportion_feu_1": float(p.feu.mean())}
        for nom, p in parties.items()
    ])


def construire_caracteristiques(X: pd.DataFrame, log_pluie: bool = False) -> pd.DataFrame:
    """Variables disponibles sans estimation : calendrier et éventuellement log1p pluie.

    L'entrée est une table brute en unités initiales, jamais un CSV déjà standardisé.
    """
    X = X[ENTREES].copy()
    dates = pd.to_datetime(X["acq_date"], errors="raise")
    periode = np.where(dates.dt.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * (dates.dt.day_of_year - 1) / periode
    X["jour_sin"] = np.sin(angle)
    X["jour_cos"] = np.cos(angle)
    saisons = {12: "hiver", 1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps", 5: "printemps",
               6: "ete", 7: "ete", 8: "ete", 9: "automne", 10: "automne", 11: "automne"}
    X["saison"] = dates.dt.month.map(saisons)
    if log_pluie:
        X["precipitation_sum"] = np.log1p(X["precipitation_sum"])
    return X[NUMERIQUES + ["saison"]]


def creer_pipeline(mise_echelle: str = "standard", log_pluie: bool = False) -> Pipeline:
    fabricant = SCALERS[mise_echelle]
    numerique = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("echelle", fabricant() if fabricant else "passthrough"),
    ])
    categoriel = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preparation = ColumnTransformer([
        ("num", numerique, NUMERIQUES), ("cat", categoriel, ["saison"]),
    ], remainder="drop")
    return Pipeline([
        ("caracteristiques", FunctionTransformer(construire_caracteristiques, validate=False,
                                               kw_args={"log_pluie": log_pluie})),
        ("preparation", preparation),
        ("modele", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ])


def mesures(y: pd.Series, probabilites: np.ndarray) -> dict:
    prediction = (probabilites >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilites)),
        "average_precision": float(average_precision_score(y, probabilites)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "rappel": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def comparer_sur_validation(parties: dict, out: Path = OUT) -> tuple[Pipeline, pd.DataFrame, str]:
    train, validation = parties["train"], parties["validation"]
    lignes, pipelines = [], {}
    configurations = [(nom, False) for nom in SCALERS] + [("standard", True), ("robust", True)]
    for nom, log in configurations:
        identifiant = nom + ("_log1p_pluie" if log else "_brut")
        pipeline = creer_pipeline(nom, log)
        pipeline.fit(train[ENTREES], train.feu)
        proba = pipeline.predict_proba(validation[ENTREES])[:, 1]
        lignes.append({"configuration": identifiant, **mesures(validation.feu, proba)})
        pipelines[identifiant] = pipeline
    comparaison = pd.DataFrame(lignes).sort_values(
        ["average_precision", "roc_auc", "configuration"], ascending=[False, False, True]
    ).reset_index(drop=True)
    choix = str(comparaison.loc[0, "configuration"])
    exporter(comparaison, "comparaison_validation", out)
    return pipelines[choix], comparaison, choix


def comparer_echelles(train: pd.DataFrame, out: Path = OUT) -> pd.DataFrame:
    """Statistiques uniquement sur le train ; les quatre scalers sont alternatifs."""
    X = SimpleImputer(strategy="median").fit_transform(train[METEO])
    lignes = []
    for nom, constructeur in SCALERS.items():
        valeurs = constructeur().fit_transform(X) if constructeur else X
        for j, col in enumerate(METEO):
            v = valeurs[:, j]
            lignes.append({"methode": nom, "variable": col, "min": v.min(), "max": v.max(),
                           "moyenne": v.mean(), "ecart_type_ddof0": v.std(ddof=0),
                           "mediane": np.median(v), "iqr": np.quantile(v, .75) - np.quantile(v, .25)})
    resultat = pd.DataFrame(lignes)
    exporter(resultat, "statistiques_echelles_train", out)
    return resultat


def comparer_repartitions(train: pd.DataFrame, out: Path = OUT) -> pd.DataFrame:
    """Comparer des partitions internes au train, distinctes du split temporel."""
    a, b = train_test_split(np.arange(len(train)), test_size=.2, stratify=train.feu, random_state=42)
    a2, b2 = train_test_split(np.arange(len(train)), test_size=.2, stratify=train.feu, random_state=42)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    lignes = [{"exemple": "80_20_stratifie_train_uniquement", "partie": nom, "lignes": len(idx),
               "proportion_feu_1": float(train.iloc[idx].feu.mean())} for nom, idx in [("80%", a), ("20%", b)]]
    for pli, (apprentissage, controle) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(train, train.feu), 1):
        lignes.append({"exemple": f"kfold_stratifie_interne_{pli}", "partie": "controle",
                       "lignes": len(controle), "proportion_feu_1": float(train.iloc[controle].feu.mean())})
    resultat = pd.DataFrame(lignes)
    exporter(resultat, "comparaisons_repartition_train", out)
    return resultat


def extensions_selection_pca(train: pd.DataFrame, out: Path = OUT) -> dict:
    """Analyse exploratoire de la sélection de variables et de la dimension."""
    p = creer_pipeline("standard")[:-1]
    X = p.fit_transform(train[ENTREES], train.feu)
    noms = p.named_steps["preparation"].get_feature_names_out()
    selecteur = SelectKBest(f_classif, k=min(6, X.shape[1])).fit(X, train.feu)
    selection = pd.DataFrame({"variable": noms, "score_F": selecteur.scores_,
                              "retenue_demo": selecteur.get_support()}).sort_values("score_F", ascending=False)
    acp = PCA(n_components=.95, svd_solver="full").fit(X)
    variance = pd.DataFrame({"composante": np.arange(1, acp.n_components_ + 1),
                             "variance_expliquee": acp.explained_variance_ratio_,
                             "variance_cumulee": np.cumsum(acp.explained_variance_ratio_)})
    exporter(selection, "extension_selection_train", out)
    exporter(variance, "extension_pca_train", out)
    return {"selection": selection, "pca": variance}


def validation_temporelle_interne(train: pd.DataFrame, out: Path = OUT) -> pd.DataFrame:
    """Deux fenêtres annuelles dans 2018–2023 ; aucun pli ne coupe une journée."""
    lignes = []
    for annee in (2022, 2023):
        avant = train.loc[train.acq_date.dt.year.lt(annee)]
        apres = train.loc[train.acq_date.dt.year.eq(annee)]
        p = creer_pipeline("standard")
        p.fit(avant[ENTREES], avant.feu)
        lignes.append({"apprentissage_fin": annee - 1, "controle_annee": annee,
                       "n_train": len(avant), "n_controle": len(apres),
                       **mesures(apres.feu, p.predict_proba(apres[ENTREES])[:, 1])})
    resultat = pd.DataFrame(lignes)
    exporter(resultat, "validation_temporelle_interne", out)
    return resultat


def finaliser(pipeline: Pipeline, parties: dict, choix: str, rapport: dict, out: Path = OUT) -> dict:
    """Un test final, pipeline brut→prédiction et exports identifiés par observation."""
    out.mkdir(parents=True, exist_ok=True)
    train, test = parties["train"], parties["test"]
    proba = pipeline.predict_proba(test[ENTREES])[:, 1]
    score = mesures(test.feu, proba)
    dummy = DummyClassifier(strategy="prior").fit(train[ENTREES], train.feu)
    reference = mesures(test.feu, dummy.predict_proba(test[ENTREES])[:, 1])
    exporter(pd.DataFrame([{"modele": choix, **score}, {"modele": "reference_frequence_train", **reference}]),
             "evaluation_test_final", out)
    matrice = confusion_matrix(test.feu, proba >= .5, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.imshow(matrice, cmap="Blues")
    for i in range(2):
        for j in range(2):
            couleur = "white" if matrice[i, j] > matrice.max() / 2 else "black"
            ax.text(j, i, str(matrice[i, j]), ha="center", va="center", fontsize=16, color=couleur)
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Classe prédite (seuil 0,5)", ylabel="Classe observée",
           title="Test 2025, modèle choisi sur 2024")
    fig.savefig(out / "matrice_confusion_test.png", dpi=150)
    plt.close(fig)
    preparation = pipeline[:-1]
    noms = pipeline.named_steps["preparation"].get_feature_names_out()
    for nom, partie in parties.items():
        transforme = preparation.transform(partie[ENTREES])
        assert np.isfinite(transforme).all()
        tableau = pd.DataFrame(transforme, columns=noms)
        tableau.insert(0, "id_observation", partie.id_observation.to_numpy())
        exporter(tableau, f"X_{nom}_pretraite", out)
        exporter(partie[["id_observation", "feu"]], f"y_{nom}", out)
    index = pd.concat([p[["id_observation", "acq_date", "lat_c", "lon_c"]].assign(ensemble=n) for n, p in parties.items()])
    exporter(index, "index_repartition", out)
    predictions = test[["id_observation", "acq_date", "feu"]].copy()
    predictions["probabilite_modele"] = proba
    predictions["prediction_seuil_05"] = (proba >= .5).astype(int)
    exporter(predictions, "predictions_test", out)
    joblib.dump(pipeline, out / "pipeline_complet.joblib", compress=3)
    # Recharger depuis le disque et vérifier une entrée brute, pour prévenir
    # l'erreur historique d'un scaler réajusté sur des entrées déjà standardisées.
    recharge = joblib.load(out / "pipeline_complet.joblib")
    np.testing.assert_allclose(recharge.predict_proba(test[ENTREES].iloc[:30])[:, 1], proba[:30], rtol=0, atol=1e-12)
    imputation = pipeline.named_steps["preparation"].named_transformers_["num"].named_steps["imputation"]
    attendues = construire_caracteristiques(train[ENTREES], **pipeline.named_steps["caracteristiques"].kw_args)[NUMERIQUES].median()
    np.testing.assert_allclose(imputation.statistics_, attendues.to_numpy())
    # Injection future extrême : transform ne doit modifier aucun paramètre appris.
    imputation_rechargee = recharge.named_steps["preparation"].named_transformers_["num"].named_steps["imputation"]
    statistiques_avant = imputation_rechargee.statistics_.copy()
    futur = test[ENTREES].iloc[:2].copy()
    futur.loc[:, "temperature_2m_max"] = 1000.0
    futur.loc[:, "humidite_moy"] = np.nan
    assert np.isfinite(recharge.predict_proba(futur)).all()
    np.testing.assert_array_equal(imputation_rechargee.statistics_, statistiques_avant)
    rapport.update({
        "repartition": bilan_repartition(parties).to_dict(orient="records"),
        "configuration_choisie_sur_2024": choix,
        "critere_selection": "average_precision validation, puis ROC-AUC en cas d'égalité",
        "seuil_prediction_fixe": .5, "evaluation_test_2025": score,
        "nombre_variables_apres_preparation": len(noms),
        "entrees_brutes_pipeline": ENTREES, "variables_modele": list(noms),
        "parametres_appris_sur": "2018-2023 uniquement, sans réentraînement sur 2024",
        "controles_execution": ["ensembles chronologiques et disjoints", "médianes égales au train seul",
                                  "exports finis", "pipeline sauvegardé/rechargé : prédictions identiques",
                                  "transform d'une entrée future avec NaN : paramètres inchangés"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
        "limites": ["Association rétrospective avec météo observée du même jour, pas une prévision J+1.",
                    "Origine des exemples feu=0 et export FIRMS brut à documenter.",
                    "Caractère forestier et localisation exacte non certifiés par cette table.",
                    "Lieux déjà présents en entraînement possibles en test : nouveaux lieux non évalués.",
                    "Probabilités non calibrées sur la prévalence réelle dans la population.",
                    "Comparaisons sur des absences simulées, sans conclusion sur le mécanisme réel de collecte."]
    })
    (out / "rapport_application.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    return rapport


def executer(out: Path = OUT) -> dict:
    from comparaisons import comparer_imputations, comparer_encodages
    from aberrants import analyser_aberrants

    donnees, rapport = charger_nettoyer()
    parties = repartir(donnees)
    train = parties["train"]
    exporter(bilan_repartition(parties), "repartition_chronologique", out)
    manquants = pd.DataFrame({"variable": donnees.columns,
                              "nombre_NaN": donnees.isna().sum().to_numpy(),
                              "proportion_NaN": donnees.isna().mean().to_numpy()})
    exporter(manquants, "manquants_reels", out)
    print(bilan_repartition(parties).to_string(index=False), flush=True)
    comparer_imputations(train, out)
    comparer_encodages(train, out)
    analyser_aberrants(train, out)
    comparer_echelles(train, out)
    comparer_repartitions(train, out)
    extensions_selection_pca(train, out)
    validation_temporelle_interne(train, out)
    pipeline, comparaison, choix = comparer_sur_validation(parties, out)
    print("Comparaison sur 2024 :\n", comparaison.to_string(index=False), flush=True)
    rapport = finaliser(pipeline, parties, choix, rapport, out)
    print("Test final 2025 :", json.dumps(rapport["evaluation_test_2025"], ensure_ascii=False), flush=True)
    return rapport


if __name__ == "__main__":
    # Stabiliser le nom de la fonction sérialisée dans joblib, même en CLI.
    import sys
    sys.path.insert(0, str(BASE))
    import pipeline_pretraitement
    pipeline_pretraitement.executer()
