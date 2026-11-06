"""Exécuter et conserver les sorties du notebook avec le kernel du Python courant.

Ce petit exécuteur utilise jupyter_client/ipykernel déjà présents dans le projet.
Il n'installe rien et ne dépend pas de nbformat ou de nbclient.
"""

from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
import argparse
import json
import os
import sys
import time

from jupyter_client import KernelManager


BASE = Path(__file__).resolve().parent


def enregistrer(notebook: dict, chemin: Path) -> None:
    chemin.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def executer(chemin: Path, timeout: float = 120) -> None:
    """Exécuter les cellules dans l'ordre, conserver HTML/images et arrêter sur erreur."""
    chemin = Path(chemin).resolve()
    notebook = json.loads(chemin.read_text(encoding="utf-8"))
    # Une erreur ne doit pas laisser de vieilles sorties dans les cellules suivantes.
    for cellule in notebook["cells"]:
        if cellule["cell_type"] == "code":
            cellule["execution_count"] = None
            cellule["outputs"] = []
    notebook.setdefault("metadata", {})["execution_preparation"] = {"statut": "en_cours"}
    enregistrer(notebook, chemin)
    debut_global = time.monotonic()

    with TemporaryDirectory(prefix="notebook-preparation-") as temporaire:
        temporaire = Path(temporaire)
        environnement = os.environ.copy()
        environnement.update({
            "PYTHONIOENCODING": "utf-8",
            "MPLBACKEND": "Agg",
            "IPYTHONDIR": str(temporaire / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(temporaire / "jupyter-runtime"),
            "MPLCONFIGDIR": str(temporaire / "matplotlib"),
        })
        km = KernelManager(kernel_name="python3", connection_file=str(temporaire / "kernel.json"))
        client = None
        demarre = False
        try:
            spec = km.kernel_spec
            if spec is None:
                raise RuntimeError("Kernel Python introuvable ; ipykernel doit être disponible dans l'environnement.")
            # Ne pas reprendre un interpréteur d'un autre kernelspec utilisateur.
            spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            km.start_kernel(cwd=str(BASE), env=environnement)
            demarre = True
            client = km.client()
            client.start_channels()
            client.wait_for_ready(timeout=60)
            total = sum(c["cell_type"] == "code" for c in notebook["cells"])
            compteur = 0
            for position, cellule in enumerate(notebook["cells"], 1):
                if cellule["cell_type"] != "code":
                    continue
                compteur += 1
                print(f"Cellule de code {compteur}/{total} (position {position})...", flush=True)
                source = cellule["source"]
                if isinstance(source, list):
                    source = "".join(source)
                message_id = client.execute(source, store_history=True, allow_stdin=False, stop_on_error=True)
                limite = time.monotonic() + timeout
                erreur = None
                effacer_avant_prochaine_sortie = False
                while True:
                    restant = limite - time.monotonic()
                    if restant <= 0:
                        raise TimeoutError(f"Cellule {position} : délai de {timeout:g} secondes dépassé.")
                    try:
                        message = client.get_iopub_msg(timeout=min(1.0, restant))
                    except Empty:
                        if not km.is_alive():
                            raise RuntimeError(f"Le kernel s'est arrêté pendant la cellule {position}.")
                        continue
                    if message.get("parent_header", {}).get("msg_id") != message_id:
                        continue
                    genre = message["header"]["msg_type"]
                    contenu = message["content"]
                    if genre == "status" and contenu.get("execution_state") == "idle":
                        break
                    if genre == "execute_input":
                        cellule["execution_count"] = contenu["execution_count"]
                    elif genre == "clear_output":
                        if contenu.get("wait", False):
                            effacer_avant_prochaine_sortie = True
                        else:
                            cellule["outputs"] = []
                    elif genre in ("stream", "display_data", "execute_result", "error"):
                        if effacer_avant_prochaine_sortie:
                            cellule["outputs"] = []
                            effacer_avant_prochaine_sortie = False
                        if genre == "stream":
                            sortie = {"output_type": "stream", "name": contenu["name"], "text": contenu["text"]}
                        elif genre in ("display_data", "execute_result"):
                            sortie = {
                                "output_type": genre,
                                "data": contenu.get("data", {}),
                                "metadata": contenu.get("metadata", {}),
                            }
                            if genre == "execute_result":
                                sortie["execution_count"] = contenu["execution_count"]
                        else:
                            sortie = {"output_type": "error", "ename": contenu["ename"],
                                      "evalue": contenu["evalue"], "traceback": contenu["traceback"]}
                            erreur = f"{contenu['ename']}: {contenu['evalue']}"
                        cellule["outputs"].append(sortie)
                # Consommer aussi la réponse shell pour détecter un échec silencieux.
                while True:
                    restant = limite - time.monotonic()
                    if restant <= 0:
                        raise TimeoutError(f"Réponse shell absente pour la cellule {position}.")
                    try:
                        reponse = client.get_shell_msg(timeout=min(1.0, restant))
                    except Empty:
                        continue
                    if reponse.get("parent_header", {}).get("msg_id") == message_id:
                        if reponse["content"].get("status") == "error" and erreur is None:
                            erreur = reponse["content"].get("evalue", "Erreur d'exécution")
                        break
                enregistrer(notebook, chemin)
                if erreur is not None:
                    raise RuntimeError(f"Échec de la cellule {position} : {erreur}")
                print(f"  {len(cellule['outputs'])} sorties enregistrées.", flush=True)

            notebook["metadata"]["language_info"]["version"] = sys.version.split()[0]
            notebook["metadata"]["execution_preparation"] = {
                "statut": "reussi", "cellules_executees": compteur,
                "duree_secondes": round(time.monotonic() - debut_global, 2),
                "interpreteur": sys.executable,
            }
            enregistrer(notebook, chemin)
            print(f"Notebook exécuté et enregistré : {chemin}", flush=True)
        except BaseException as exception:
            notebook["metadata"]["execution_preparation"] = {
                "statut": "erreur", "message": str(exception),
                "duree_secondes": round(time.monotonic() - debut_global, 2),
            }
            enregistrer(notebook, chemin)
            raise
        finally:
            try:
                if demarre:
                    km.shutdown_kernel(now=True)
            finally:
                if client is not None:
                    client.stop_channels()
                if not demarre:
                    km.cleanup_resources()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("notebook", nargs="?", type=Path,
                         default=BASE.parent / "Analyse_et_Modelisation" / "04_Preparation_Donnees.ipynb")
    options.add_argument("--timeout", type=float, default=120, help="Délai maximal par cellule, en secondes.")
    args = options.parse_args()
    if args.timeout <= 0:
        options.error("--timeout doit être strictement positif")
    executer(args.notebook, timeout=args.timeout)
