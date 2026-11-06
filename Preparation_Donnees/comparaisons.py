"""Comparaisons des méthodes à partir de l'ensemble d'entraînement.

Les données manquantes de ces analyses sont simulées sur des copies. Les
transformations présentées ici ne modifient ni le fichier source ni le pipeline
retenu pour la préparation réelle du projet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


METEO = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "humidite_moy",
]
SAISONS = {
    12: "hiver", 1: "hiver", 2: "hiver",
    3: "printemps", 4: "printemps", 5: "printemps",
    6: "été", 7: "été", 8: "été",
    9: "automne", 10: "automne", 11: "automne",
}


def _saison(frame: pd.DataFrame) -> pd.Series:
    """Dériver une saison calendaire sans apprendre de paramètres sur les données."""
    if "mois" in frame.columns:
        mois = pd.to_numeric(frame["mois"], errors="raise")
    else:
        mois = pd.to_datetime(frame["acq_date"], errors="raise").dt.month
    saisons = mois.map(SAISONS)
    if saisons.isna().any():
        raise ValueError("Les mois doivent être valides, compris entre 1 et 12.")
    return saisons.rename("saison")


def _exporter(tables: dict[str, pd.DataFrame], out: Path, prefixe: str) -> None:
    """Enregistrer les tableaux comparatifs sans exporter d'objets entraînés."""
    out.mkdir(parents=True, exist_ok=True)
    for nom, table in tables.items():
        table.to_csv(out / f"{prefixe}_{nom}.csv", index=False, encoding="utf-8-sig")


def comparer_imputations(train: pd.DataFrame, out: Path) -> dict:
    """Comparer six imputations sur des absences MCAR artificielles.

    Au maximum 1 200 lignes complètes du seul ensemble d'entraînement servent
    à cette comparaison. Un sous-échantillon de 80 % fournit les donneurs, et les 20 %
    restants servent à l'évaluation interne. Environ 5 % des cellules de
    chaque variable sont masquées dans chacun de ces deux groupes, avec une
    graine fixée à 42. Les imputateurs et les scalers sont ajustés uniquement
    sur les donneurs masqués. Les valeurs vraies du groupe d'évaluation ne
    participent jamais au fit et servent uniquement au calcul des erreurs.

    Les MAE et RMSE sont calculées séparément pour chaque variable, dans son
    unité d'origine. Aucun calcul de cette comparaison n'utilise la cible ``feu``.
    """
    if not train.index.is_unique:
        raise ValueError("Réinitialiser l'index du train avant la comparaison : index unique requis.")
    rng = np.random.default_rng(42)
    disponibles = train.loc[:, METEO].apply(pd.to_numeric, errors="raise")
    disponibles = disponibles.replace([np.inf, -np.inf], np.nan).dropna()
    if len(disponibles) < 20:
        raise ValueError("Il faut au moins 20 lignes météo complètes pour cette comparaison.")
    taille = min(1200, len(disponibles))
    echantillon = disponibles.sample(n=taille, random_state=42).copy()
    permutation = rng.permutation(taille)
    coupe = int(taille * 0.8)
    donneurs = echantillon.iloc[permutation[:coupe]].copy()
    evaluation = echantillon.iloc[permutation[coupe:]].copy()

    def masquer(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        masque = np.zeros(frame.shape, dtype=bool)
        nombre = max(1, round(0.05 * len(frame)))
        for colonne in range(frame.shape[1]):
            masque[rng.choice(len(frame), size=nombre, replace=False), colonne] = True
        return frame.mask(masque), masque

    donneurs_masques, masque_fit = masquer(donneurs)
    evaluation_masquee, masque_eval = masquer(evaluation)
    strategies = {
        "constante_0": SimpleImputer(strategy="constant", fill_value=0),
        "moyenne": SimpleImputer(strategy="mean"),
        "médiane": SimpleImputer(strategy="median"),
        "mode": SimpleImputer(strategy="most_frequent"),
    }
    predictions = {}
    for nom, imputer in strategies.items():
        imputer.fit(donneurs_masques)
        predictions[nom] = imputer.transform(evaluation_masquee)

    # StandardScaler accepte les NaN et les conserve pour les imputateurs.
    scaler = StandardScaler().fit(donneurs_masques)
    donneurs_std = scaler.transform(donneurs_masques)
    evaluation_std = scaler.transform(evaluation_masquee)
    imputeurs_relations = {
        "KNN_standardisé": KNNImputer(n_neighbors=3, weights="uniform"),
        "régression_itérative_standardisée": IterativeImputer(
            estimator=BayesianRidge(), max_iter=20, tol=1e-3,
            initial_strategy="median", random_state=42,
        ),
    }
    for nom, imputer in imputeurs_relations.items():
        imputer.fit(donneurs_std)
        predictions[nom] = scaler.inverse_transform(imputer.transform(evaluation_std))

    scores = []
    verite = evaluation.to_numpy(dtype=float)
    for methode, imputation in predictions.items():
        for j, variable in enumerate(METEO):
            differences = imputation[masque_eval[:, j], j] - verite[masque_eval[:, j], j]
            scores.append({
                "méthode": methode,
                "variable": variable,
                "cellules_évaluées": int(masque_eval[:, j].sum()),
                "MAE": float(np.mean(np.abs(differences))),
                "RMSE": float(np.sqrt(np.mean(differences ** 2))),
            })

    comparaison = pd.DataFrame(scores)
    masquage = pd.DataFrame({
        "variable": METEO,
        "lignes_donneurs": len(donneurs),
        "absences_donneurs": masque_fit.sum(axis=0),
        "lignes_évaluation": len(evaluation),
        "absences_évaluation": masque_eval.sum(axis=0),
        "taux_absences_évaluation": masque_eval.mean(axis=0),
    })

    # Suppressions démontrées sur une copie, sans retirer d'observation réelle.
    copie = pd.concat([donneurs_masques, evaluation_masquee], ignore_index=True)
    apres_lignes = copie.dropna()
    copie_colonnes = copie.copy()
    copie_colonnes["capteur_demo_tres_incomplet"] = 1.0
    trop_incomplet = rng.choice(len(copie_colonnes), size=round(0.7 * len(copie_colonnes)), replace=False)
    copie_colonnes.loc[trop_incomplet, "capteur_demo_tres_incomplet"] = np.nan
    taux = copie_colonnes.isna().mean()
    colonnes_a_retirer = taux[taux > 0.5].index.tolist()
    apres_colonnes = copie_colonnes.drop(columns=colonnes_a_retirer)
    suppression = pd.DataFrame([
        {
            "opération": "dropna sur les lignes artificiellement incomplètes",
            "avant": len(copie), "après": len(apres_lignes),
            "éléments_retirés": len(copie) - len(apres_lignes),
            "unité": "lignes", "colonnes_retirées": "",
        },
        {
            "opération": "suppression des colonnes avec plus de 50 % de NaN",
            "avant": copie_colonnes.shape[1], "après": apres_colonnes.shape[1],
            "éléments_retirés": len(colonnes_a_retirer),
            "unité": "colonnes", "colonnes_retirées": ", ".join(colonnes_a_retirer),
        },
    ])

    # Variable catégorielle dérivée du calendrier. Son mode vient des donneurs.
    saisons_donneurs = _saison(train.loc[donneurs.index]).astype(object)
    saisons_evaluation = _saison(train.loc[evaluation.index]).astype(object)
    saison_masquee = saisons_evaluation.copy()
    positions = rng.choice(len(saison_masquee), size=max(1, round(0.05 * len(saison_masquee))), replace=False)
    saison_masquee.iloc[positions] = np.nan
    mode = saisons_donneurs.mode().iloc[0]
    categories = pd.DataFrame({
        "saison_originale": saisons_evaluation.to_numpy(),
        "saison_avec_absences_artificielles": saison_masquee.to_numpy(),
        "imputation_mode": saison_masquee.fillna(mode).to_numpy(),
        "imputation_catégorie_spécifique": saison_masquee.fillna("Inconnu").to_numpy(),
    })
    # Les absences apparaissent en tête, pour rendre l'exemple lisible.
    categories["absence_artificielle"] = saison_masquee.isna().to_numpy()
    categories = categories.sort_values("absence_artificielle", ascending=False).head(30)

    tables = {
        "comparaison": comparaison,
        "masquage": masquage,
        "suppression": suppression,
        "categories": categories,
    }
    _exporter(tables, Path(out), "comparaison_imputation")
    note = (
        "Absences artificielles MCAR sur des copies : environ 5 % par variable, graine 42. "
        "Les 20 % réservés à l'évaluation interne appartiennent exclusivement au train "
        "du projet. Leurs vérités masquées ne participent à aucun ajustement. Les MAE/RMSE "
        "portent uniquement sur ces cellules, séparément par variable et dans les unités "
        "d'origine. MCAR signifie une absence indépendante des valeurs ; MAR une absence "
        "conditionnelle aux variables observées ; MNAR une dépendance résiduelle à la valeur "
        "non observée. Le mécanisme réel ne peut pas être déduit de cette simulation ou de "
        "l'absence actuelle de NaN. Les seuils de suppression sont des repères à "
        "adapter au contexte. Aucun de ces essais n'impose une imputation au dataset réel."
    )
    return {**tables, "note": note}


def comparer_encodages(train: pd.DataFrame, out: Path) -> dict:
    """Illustrer OneHotEncoder, LabelEncoder et OrdinalEncoder sur des copies.

    OneHotEncoder est ajusté sur la saison du train. LabelEncoder concerne
    uniquement une copie textuelle de la cible. OrdinalEncoder illustre un
    ordre défini explicitement sur des classes exploratoires d'humidité.
    La comparaison est indépendante du pipeline final et ne change pas ``feu``.
    """
    saisons = _saison(train).to_frame()
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot.fit(saisons)
    exemple_saisons = saisons.drop_duplicates().reset_index(drop=True)
    exemples_onehot = pd.concat([
        exemple_saisons,
        pd.DataFrame(onehot.transform(exemple_saisons), columns=onehot.get_feature_names_out()),
    ], axis=1)
    inconnue = pd.DataFrame({"saison": ["saison_inconnue_demo"]})
    exemple_inconnu = pd.concat([
        inconnue,
        pd.DataFrame(onehot.transform(inconnue), columns=onehot.get_feature_names_out()),
    ], axis=1)

    cible_originale = pd.to_numeric(train["feu"], errors="raise")
    if not cible_originale.isin([0, 1]).all():
        raise ValueError("La cible doit être codée 0/1.")
    cible_texte = cible_originale.map({0: "absence_detection", 1: "detection"})
    label = LabelEncoder().fit(cible_texte)
    exemples_label = pd.DataFrame({
        "feu_original": cible_originale.to_numpy(),
        "cible_texte_demo": cible_texte.to_numpy(),
        "LabelEncoder_demo": label.transform(cible_texte),
    }).drop_duplicates().sort_values("feu_original").reset_index(drop=True)
    classes_label = pd.DataFrame({
        "classe": label.classes_,
        "code": label.transform(label.classes_),
    })

    humidite = pd.to_numeric(train["humidite_moy"], errors="raise")
    if humidite.isna().any() or not humidite.between(0, 100).all():
        raise ValueError("L'humidité doit être présente et comprise entre 0 et 100.")
    niveaux = ["faible", "moyenne", "élevée"]
    humidite_classe = pd.cut(
        humidite, bins=[-np.inf, 30, 60, np.inf], labels=niveaux,
        right=True,
    ).astype(object).to_frame(name="classe_humidite_demo")
    ordinal = OrdinalEncoder(categories=[niveaux], handle_unknown="use_encoded_value", unknown_value=-1)
    ordinal.fit(humidite_classe)
    exemples_ordinal = pd.DataFrame({
        "humidite_moy_originale": humidite.to_numpy(),
        "classe_humidite_demo": humidite_classe.iloc[:, 0].to_numpy(),
        "OrdinalEncoder_demo": ordinal.transform(humidite_classe)[:, 0],
    }).groupby("classe_humidite_demo", sort=False).head(5).reset_index(drop=True)

    tables = {
        "onehot": exemples_onehot,
        "onehot_inconnu": exemple_inconnu,
        "label": exemples_label,
        "classes_label": classes_label,
        "ordinal": exemples_ordinal,
    }
    _exporter(tables, Path(out), "comparaison_encodage")
    note = (
        "Comparaison hors pipeline final. OneHotEncoder représente des saisons sans ordre "
        "numérique et ignore une catégorie inconnue en produisant des zéros. "
        "LabelEncoder code une copie textuelle de la cible ; feu conserve ses valeurs "
        "originales 0/1. LabelEncoder ne définit pas l'ordre métier d'une variable "
        "explicative. OrdinalEncoder reçoit explicitement l'ordre faible < moyenne < élevée. "
        "Les seuils d'humidité 30 % et 60 % servent seulement à l'illustration : faible "
        "pour humidité ≤ 30 %, moyenne pour 30 % < humidité ≤ 60 %, élevée au-delà. "
        "Ils ne constituent pas une échelle validée du risque d'incendie."
    )
    return {**tables, "note": note}
