"""Diagnostics des valeurs atypiques, appris sur le train seulement.

Un signal statistique n'établit pas qu'une mesure est fausse. Les données réelles
restent intactes : écrêtage et transformations sont des comparaisons sur copies.
"""

from pathlib import Path

import numpy as np
import pandas as pd


METEO = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "humidite_moy",
]
LIBELLES = {
    "temperature_2m_max": "Température maximale (°C)",
    "precipitation_sum": "Précipitations (mm)",
    "wind_speed_10m_max": "Vent maximal (km/h)",
    "humidite_moy": "Humidité moyenne (%)",
}


def _statistiques(donnees: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Mesure l'asymétrie et les étendues, sans modifier le tableau reçu."""
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "variable": colonne,
                "effectif": len(donnees),
                "minimum": float(donnees[colonne].min()),
                "maximum": float(donnees[colonne].max()),
                "moyenne": float(donnees[colonne].mean()),
                "ecart_type_population": float(donnees[colonne].std(ddof=0)),
                "asymetrie_skew": float(donnees[colonne].skew()),
            }
            for colonne in donnees.columns
        ]
    )


def analyser_aberrants(train: pd.DataFrame, out: Path) -> dict[str, pd.DataFrame]:
    """Exporte des diagnostics univariés/multivariés et deux figures scientifiques.

    ``train`` doit contenir ``id_observation``, ``feu`` et les quatre variables
    météo numériques complètes. La fonction ne reçoit ni validation, ni test.
    Les seuils, l'échantillonnage et les modèles utilisent uniquement ce train.
    """
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    requis = ["id_observation", "feu", *METEO]
    absentes = sorted(set(requis).difference(train.columns))
    if absentes:
        raise ValueError(f"Colonnes absentes du train : {absentes}")
    if len(train) < 3 or not train["id_observation"].is_unique:
        raise ValueError("Il faut au moins trois observations avec identifiants uniques.")
    valeurs = train[METEO].astype(float).copy()
    if not np.isfinite(valeurs.to_numpy()).all():
        raise ValueError("Imputer/contrôler les valeurs météo avant ce diagnostic.")
    if valeurs[["precipitation_sum", "wind_speed_10m_max"]].lt(0).any().any():
        raise ValueError("Pluie et vent doivent être non négatifs pour log1p/racine.")

    moyenne = valeurs.mean()
    ecart_type = valeurs.std(ddof=0)
    # Une variable constante a un Z-score défini ici à zéro.
    zscore = ((valeurs - moyenne) / ecart_type.replace(0, np.nan)).fillna(0.0)
    q1 = valeurs.quantile(0.25)
    q3 = valeurs.quantile(0.75)
    iqr = q3 - q1
    bas = q1 - 1.5 * iqr
    haut = q3 + 1.5 * iqr
    flags_iqr = valeurs.lt(bas) | valeurs.gt(haut)
    flags_z = zscore.abs().gt(3)
    zone_z = zscore.abs().ge(2) & zscore.abs().le(3)

    bornes = pd.DataFrame(
        {
            "variable": METEO,
            "n_train": len(train),
            "moyenne_train": moyenne.reindex(METEO).to_numpy(),
            "ecart_type_train_ddof0": ecart_type.reindex(METEO).to_numpy(),
            "q1_train": q1.reindex(METEO).to_numpy(),
            "q3_train": q3.reindex(METEO).to_numpy(),
            "iqr_train": iqr.reindex(METEO).to_numpy(),
            "borne_iqr_basse": bas.reindex(METEO).to_numpy(),
            "borne_iqr_haute": haut.reindex(METEO).to_numpy(),
            "n_signal_z_sup3": flags_z.sum().reindex(METEO).to_numpy(),
            "n_zone_z_2_a_3": zone_z.sum().reindex(METEO).to_numpy(),
            "n_signal_iqr": flags_iqr.sum().reindex(METEO).to_numpy(),
            "iqr_nul": iqr.eq(0).reindex(METEO).to_numpy(),
        }
    )
    bornes["interpretation"] = np.where(
        bornes["iqr_nul"],
        "IQR nul : toute valeur différente des quartiles peut être signalée. "
        "Ce critère peut être inadapté à une variable avec beaucoup de zéros.",
        "Signal statistique à examiner, pas une preuve d'erreur.",
    )

    flags = train[["id_observation"]].copy()
    for colonne in METEO:
        flags[f"zscore_{colonne}"] = zscore[colonne]
        flags[f"signal_z_sup3_{colonne}"] = flags_z[colonne]
        flags[f"zone_z_2_a_3_{colonne}"] = zone_z[colonne]
        flags[f"signal_iqr_{colonne}"] = flags_iqr[colonne]
    flags["signal_z_au_moins_une_variable"] = flags_z.any(axis=1)
    flags["signal_iqr_au_moins_une_variable"] = flags_iqr.any(axis=1)

    cap_bas = bas.copy()
    cap_haut = haut.copy()
    for colonne in ("precipitation_sum", "wind_speed_10m_max", "humidite_moy"):
        cap_bas[colonne] = max(0.0, cap_bas[colonne])
    cap_haut["humidite_moy"] = min(100.0, cap_haut["humidite_moy"])
    ecretee = valeurs.clip(lower=cap_bas, upper=cap_haut, axis=1)
    bornes["borne_capping_basse"] = cap_bas.reindex(METEO).to_numpy()
    bornes["borne_capping_haute"] = cap_haut.reindex(METEO).to_numpy()
    bornes["n_modifiees_capping_simule"] = (
        ecretee.ne(valeurs).sum().reindex(METEO).to_numpy()
    )
    transformations = [
        _statistiques(valeurs, "original_train"),
        _statistiques(ecretee, "capping_iqr_copie_exploratoire"),
    ]
    for nom, operation in (("log1p", np.log1p), ("racine_carree", np.sqrt)):
        copie = valeurs[["precipitation_sum", "wind_speed_10m_max"]].copy()
        copie = operation(copie)
        transformations.append(_statistiques(copie, f"{nom}_copie_exploratoire"))
    stats_transformations = pd.concat(transformations, ignore_index=True)
    scenarios_suppression = pd.DataFrame(
        [
            {
                "scenario": "suppression_si_signal_iqr",
                "n_train_initial": len(train),
                "n_exclues_si_applique": int(flags_iqr.any(axis=1).sum()),
                "n_restantes_si_applique": int((~flags_iqr.any(axis=1)).sum()),
                "applique_aux_donnees_reelles": False,
            },
            {
                "scenario": "suppression_si_signal_z_sup3",
                "n_train_initial": len(train),
                "n_exclues_si_applique": int(flags_z.any(axis=1).sum()),
                "n_restantes_si_applique": int((~flags_z.any(axis=1)).sum()),
                "applique_aux_donnees_reelles": False,
            },
        ]
    )

    # Coût borné pour LOF/KNN ; ni feu ni id_observation n'entrent dans les modèles.
    echantillon = train.sample(n=min(3000, len(train)), random_state=42).copy()
    scaler = StandardScaler()
    z = scaler.fit_transform(echantillon[METEO])
    isolation = IsolationForest(
        n_estimators=150, contamination="auto", random_state=42, n_jobs=1
    ).fit(z)
    n_voisins = min(20, len(echantillon) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_voisins, contamination="auto")
    signal_lof = lof.fit_predict(z) == -1
    # Requête explicite sur z : le premier voisin est soi-même, à distance nulle.
    k = min(5, len(echantillon) - 1)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)
    distance_knn = distances[:, -1]
    seuil_knn = float(np.quantile(distance_knn, 0.95))
    multivarie = echantillon[["id_observation", *METEO]].copy()
    multivarie["score_isolation_normalite"] = isolation.decision_function(z)
    multivarie["signal_isolation_forest"] = isolation.predict(z) == -1
    multivarie["score_lof_anomalie"] = -lof.negative_outlier_factor_
    multivarie["signal_lof"] = signal_lof
    multivarie["distance_au_k_ieme_voisin"] = distance_knn
    multivarie["k_knn_hors_soi_meme"] = k
    multivarie["seuil_knn_quantile95_train_echantillon"] = seuil_knn
    multivarie["signal_knn_exploratoire"] = distance_knn > seuil_knn
    synthese_multivarie = pd.DataFrame(
        [
            {
                "methode": nom,
                "n_train_total": len(train),
                "n_echantillon_train": len(echantillon),
                "seed_echantillon": 42,
                "n_signalees": int(multivarie[colonne].sum()),
                "interpretation": "Atypie relative à cet échantillon, à examiner.",
            }
            for nom, colonne in (
                ("IsolationForest_contamination_auto", "signal_isolation_forest"),
                (f"LOF_{n_voisins}_voisins_contamination_auto", "signal_lof"),
                ("KNN_seuil_quantile_95_exploratoire", "signal_knn_exploratoire"),
            )
        ]
    )

    fig = plt.figure(figsize=(13, 12), layout="constrained")
    grille = fig.add_gridspec(3, 2)
    for position, colonne in enumerate(METEO):
        ax = fig.add_subplot(grille[position // 2, position % 2])
        ax.boxplot(valeurs[colonne], orientation="horizontal", flierprops={"markersize": 2})
        ax.set_xlabel(LIBELLES[colonne])
        ax.set_yticks([])
        ax.set_title(f"{int(flags_iqr[colonne].sum())} signaux IQR à examiner")
        ax.grid(axis="x", alpha=0.2)
    ax = fig.add_subplot(grille[2, :])
    for label, couleur, legende in ((0, "#357f9c", "Étiquette feu = 0"), (1, "#cb6335", "Étiquette feu = 1")):
        masque = echantillon["feu"].eq(label)
        ax.scatter(
            echantillon.loc[masque, "temperature_2m_max"],
            echantillon.loc[masque, "humidite_moy"],
            s=10, alpha=0.35, color=couleur, label=legende, rasterized=True,
        )
    ax.set_xlabel(LIBELLES["temperature_2m_max"])
    ax.set_ylabel(LIBELLES["humidite_moy"])
    ax.set_title(f"Vue bivariée, échantillon train de {len(echantillon):,} lignes")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.suptitle("Valeurs atypiques : diagnostics sur l'entraînement uniquement", fontsize=15)
    fig.savefig(out / "aberrants_boxplots_et_dispersion.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    pluie = valeurs["precipitation_sum"]
    for ax, serie, titre, xlabel in (
        (axes[0], pluie, "Distribution brute", "Pluie (mm)"),
        (axes[1], np.log1p(pluie), "Transformation log1p", "log(1 + pluie en mm)"),
        (axes[2], np.sqrt(pluie), "Transformation racine carrée", "Racine de la pluie en mm"),
    ):
        ax.hist(serie, bins=45, color="#357f9c", edgecolor="white", linewidth=0.3)
        ax.set_title(titre)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nombre d'observations train")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Précipitations : alternatives comparées sur des copies", fontsize=14)
    fig.savefig(out / "aberrants_histogrammes_pluie.png", dpi=140)
    plt.close(fig)

    tables = {
        "bornes_aberrants_train": bornes,
        "flags_aberrants_train": flags,
        "stats_transformations_aberrants": stats_transformations,
        "suppression_aberrants_scenarios": scenarios_suppression,
        "aberrants_multivaries_echantillon_train": multivarie,
        "synthese_aberrants_multivaries": synthese_multivarie,
    }
    for nom, tableau in tables.items():
        tableau.to_csv(out / f"{nom}.csv", index=False, encoding="utf-8-sig")
    return tables
