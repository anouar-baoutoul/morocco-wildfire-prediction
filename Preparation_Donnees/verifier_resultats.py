"""Vérifier les résultats exportés et l'inférence depuis un nouveau processus.

Exécuter après pipeline_pretraitement.py ou après toutes les cellules du notebook.
Ce contrôle indépendant empêche notamment de distribuer des matrices et un
scaler incompatibles, ou des cibles décalées par rapport aux observations.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

import pipeline_pretraitement as pc


def verifier(out: Path = pc.OUT) -> dict:
    rapport = json.loads((out / "rapport_application.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(pc.SOURCE.read_bytes()).hexdigest() == rapport["sha256_source"], "Source modifiée"
    source, _ = pc.charger_nettoyer()
    source = source.set_index("id_observation", drop=False)
    pipeline = joblib.load(out / "pipeline_complet.joblib")
    index = pd.read_csv(out / "index_repartition.csv", parse_dates=["acq_date"])
    assert index.id_observation.is_unique
    assert set(index.id_observation) == set(source.id_observation)
    comptes = {}
    for nom, groupe in index.groupby("ensemble"):
        X = pd.read_csv(out / f"X_{nom}_pretraite.csv")
        y = pd.read_csv(out / f"y_{nom}.csv")
        np.testing.assert_array_equal(X.id_observation, y.id_observation)
        np.testing.assert_array_equal(X.id_observation, groupe.id_observation)
        brut = source.loc[X.id_observation]
        np.testing.assert_array_equal(y.feu, brut.feu)
        transforme = pipeline[:-1].transform(brut[pc.ENTREES])
        assert list(X.columns[1:]) == list(pipeline.named_steps["preparation"].get_feature_names_out())
        np.testing.assert_allclose(X.iloc[:, 1:].to_numpy(), transforme, rtol=1e-10, atol=1e-12)
        assert "feu" not in X.columns
        assert "id_observation" not in pipeline.named_steps["preparation"].get_feature_names_out()
        comptes[nom] = len(X)
    predictions = pd.read_csv(out / "predictions_test.csv")
    brut_test = source.loc[predictions.id_observation, pc.ENTREES]
    np.testing.assert_allclose(pipeline.predict_proba(brut_test)[:, 1], predictions.probabilite_modele,
                               rtol=1e-10, atol=1e-12)
    # L'API utilise des noms de colonnes : réordonner les entrées doit conserver
    # les prédictions. Une donnée manquante est traitée sans nouvel ajustement.
    exemple = brut_test.iloc[:10].copy()
    np.testing.assert_allclose(pipeline.predict_proba(exemple), pipeline.predict_proba(exemple[pc.ENTREES[::-1]]))
    exemple.loc[:, "humidite_moy"] = np.nan
    assert np.isfinite(pipeline.predict_proba(exemple)).all()
    notebook = pc.BASE.parent / "Analyse_et_Modelisation" / "04_Preparation_Donnees.ipynb"
    cellules_executees = 0
    if notebook.exists():
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert all(c.get("execution_count") is not None for c in codes), "Notebook non exécuté intégralement"
        assert not any(o.get("output_type") == "error" for c in codes for o in c.get("outputs", []))
        cellules_executees = len(codes)
    bilan = {"statut": "OK", "lignes_verifiees": comptes,
             "matrices_et_cibles_alignees": True, "inference_brute_et_exports_identiques": True,
             "ordre_colonnes_indifferent": True, "source_inchangee": True,
             "cellules_notebook_executees_sans_erreur": cellules_executees}
    (out / "verification_resultats.json").write_text(json.dumps(bilan, ensure_ascii=False, indent=2), encoding="utf-8")
    return bilan


if __name__ == "__main__":
    print(json.dumps(verifier(), ensure_ascii=False, indent=2))
