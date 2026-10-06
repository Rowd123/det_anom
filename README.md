# Détection d'anomalies sur des données PMU

Ce projet réunit trois méthodes autonomes dans un seul répertoire de travail.
Chaque méthode possède son code, ses configurations, ses tests et son mode d'emploi.

| Répertoire | Contenu | Documentation |
|---|---|---|
| `isolation_forest/` | Prétraitement, fenêtres, caractéristiques `wf_*`, Isolation Forest, seuil POT, analyse et SHAP | [README](isolation_forest/README.md) |
| `rpad_detection/` | Détection d'événements à partir de la différence de phase entre deux PMU, flux et rapports | [README](rpad_detection/README.md) |
| `omnis/` | OmniAnomaly : entraînement, reprise, inférence, reconstruction et SPOT | [README](omnis/README.md) |

Les sources exactes sont enregistrées dans [SOURCES.json](SOURCES.json).
Les changements de structure et les règles de reprise des modèles sont détaillés
dans [MIGRATION.md](MIGRATION.md). Les vérifications réalisées sont dans
[VALIDATION.md](VALIDATION.md).

## Installation

Ouvrir un terminal dans `det_anom`. Utiliser un environnement neuf
pour éviter les collisions avec les anciennes installations de `spec_anom` et
`PMU-simulator`. Le code déclare Python >= 3.10 ; la version utilisée pour la
validation est précisée dans `VALIDATION.md`.

Linux / macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell, sans devoir activer l'environnement :

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Les commandes `spectral-anomaly`, `rpad-detection`, `omni-run`, `omni-spot` et
`omni-reconstruct` sont installées dans `.venv/bin/` sous Linux, ou
`.venv\Scripts\` sous Windows. Si l'environnement n'est pas activé, utiliser leur
chemin complet. Dans les README, `python` désigne le Python de cet environnement.

Pour n'installer qu'une méthode :

```bash
python -m pip install -e ./isolation_forest
python -m pip install -e ./rpad_detection
python -m pip install -e ./omnis
```

Exécuter uniquement la ou les lignes nécessaires. PyTorch/Pyro sont requis par
OmniAnomaly, pas par la forêt CPU ni RPAD. Les dépendances de chaque méthode sont
déclarées dans son `pyproject.toml` ; `requirements.txt` les réunit sans recopier
les listes de dépendances.

SHAP et tests :

```bash
python -m pip install -e "./isolation_forest[shap,test]" -e "./rpad_detection[test]" -e "./omnis[test]"
python -m pytest
```

L'installation ne fournit aucun modèle entraîné ni aucune mesure de terrain.
Copier les modèles et données existants dans les emplacements de son choix, puis
indiquer leurs chemins. Le dossier `data/` et les dossiers de résultats sont créés
par l'utilisateur ou les commandes ; ils ne font pas partie du code livré.

## Quelle commande utiliser ?

| Besoin | Commande principale |
|---|---|
| Préparer les fenêtres pour Isolation Forest | `spectral-anomaly prepare-windows` puis `extract-windows` |
| Appliquer une forêt existante | `spectral-anomaly score-iforest` |
| Calibrer un seuil sur ses scores | `spectral-anomaly spot` |
| Afficher les fenêtres détectées sur le signal | `spectral-anomaly plot-spot` |
| Classer les scores et voir les corrélations | `spectral-anomaly analyze-scores` |
| Expliquer une forêt existante | `spectral-anomaly-shap` |
| Détecter depuis deux angles PMU synchronisés | `rpad-detection detect-phase` |
| Détecter au fil des échantillons | `rpad-detection phase-stream` |
| Entraîner ou reprendre OmniAnomaly | `omni-run --run-mode scratch` ou `resume` |
| Évaluer un checkpoint sans entraînement | `omni-run --run-mode evaluate --checkpoint ...` |
| Scorer des fenêtres complètes avec SPOT | `omni-spot --checkpoint ...` |
| Tracer les reconstructions d'un checkpoint | `omni-reconstruct --checkpoint ...` |

Toutes les commandes détaillées dans les trois README sont données **depuis la
racine `det_anom`**. Les lignes des exemples sont volontairement sur une
seule ligne pour être copiables dans Bash et PowerShell. Les redirections de flux
propres à chaque terminal sont indiquées dans le README RPAD.

## Conventions à conserver

- La normalisation des signaux et la représentation des caractéristiques restent
  propres à chaque méthode. Les trois formats d'entrée ne sont pas interchangeables.
- Le score Isolation Forest est `-score_samples`, pas une probabilité. Son module
  `spot` conserve un **seuil POT fixe** après calibration.
- `omni-spot` utilise une log-vraisemblance négative moyenne sur la fenêtre et un
  **SPOT dynamique**. `omni-run` et `omni-reconstruct` conservent leurs conventions
  historiques au dernier point ; ne pas mélanger leurs scores/seuils.
- Conserver les manifestes JSON avec les Parquet, les configurations exactes avec
  les modèles, et les versions de bibliothèques ayant servi à leur création.
- Les configurations fournies sont des exemples. En particulier, les constantes
  `mu`/`sigma` de la forêt restent à renseigner ; elles ne sont jamais déduites
  automatiquement du signal.

## Démonstrations sans données privées

```bash
python isolation_forest/examples/make_demo.py --output artifacts/if_demo
python rpad_detection/scripts/demo_phase_detection.py --output artifacts/rpad_demo
```

La première commande génère un CSV et des configurations, à utiliser avec la chaîne
décrite dans le README Isolation Forest. La deuxième exécute aussi la détection
et produit `artifacts/rpad_demo/report/index.html`. Les jeux synthétiques servent
à vérifier l'exécution, pas à mesurer les performances sur un réseau électrique.
