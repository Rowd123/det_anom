# Isolation Forest sur des fenêtres PMU

Toutes les commandes ci-dessous sont exécutées depuis **`det_anom/`**.
Installation indépendante : `python -m pip install -e ./isolation_forest`.

La chaîne est : CSV → fenêtres normalisées → caractéristiques → forêt sauvegardée
→ scores → seuil POT → décisions. Chaque étape relit les artefacts de la précédente.
Pour un modèle existant, commencer par la section « Nouveau signal avec une forêt
déjà entraînée » ; il n'est pas nécessaire de réentraîner la forêt.

## Entrées et configurations

Le CSV contient une colonne de dates, une colonne numérique et, facultativement,
une colonne de qualité. Par exemple :

```csv
DATE_MESURE;VIGY_1:VA_V1:MAG;QF
2026-09-01T00:00:00Z;240001;NORMAL
2026-09-01T00:00:00.020Z;239998;NORMAL
```

La commande traite un canal à la fois. Les dates sont lues en UTC et le CSV est
trié ; le premier doublon est conservé. La grille temporelle doit correspondre
exactement à `sampling_period`. La qualité, les petites lacunes, les exclusions et
les fenêtres incomplètes suivent les règles décrites dans
[CONFIGURATION.md](docs/CONFIGURATION.md).

| Fichier | À configurer |
|---|---|
| `isolation_forest/configs/window_preprocessing.json` | Cadence, taille/recouvrement, qualité, exclusions, constantes `mu` et `sigma` par canal |
| `isolation_forest/configs/window_features.json` | Statistiques, bandes fréquentielles, retards, variations historiques |
| `isolation_forest/configs/window_models.json` | Caractéristiques utilisées, éventuelle mise à l'échelle des caractéristiques, découpage, paramètres IF, graine |
| `isolation_forest/configs/spot.json` | `q`, quantile initial et minimum d'excès pour le seuil POT |

**Renseigner `normalization.V1.MAG.mu` et `.sigma` avant le prétraitement.** Les
valeurs `null` de l'exemple provoquent volontairement une erreur. La formule est
`(x - mu) / sigma`, avec des constantes choisies par l'utilisateur et `sigma > 0`.
Le nom `V1.MAG` doit correspondre au `--channel-id` de la commande.

L'exemple standard utilise 50 Hz, 3000 échantillons par fenêtre (60 s) et 1500 de
recouvrement (pas de 30 s). Une modification de ces valeurs doit rester compatible
avec les bandes fréquentielles et retards configurés.

## Première utilisation : apprentissage et évaluation

Après adaptation des configurations et du chemin du CSV :

```bash
spectral-anomaly prepare-windows data/pmu.csv artifacts/if/windows.parquet --config isolation_forest/configs/window_preprocessing.json --index-col DATE_MESURE --value-col VIGY_1:VA_V1:MAG --source-id VIGY_1 --channel-id V1.MAG --sep ";"
spectral-anomaly extract-windows artifacts/if/windows.parquet artifacts/if/features.parquet --config isolation_forest/configs/window_features.json
spectral-anomaly split-windows artifacts/if/features.parquet artifacts/if/splits --config isolation_forest/configs/window_models.json
spectral-anomaly train-iforest artifacts/if/splits/train.parquet artifacts/if/window_iforest.pkl --config isolation_forest/configs/window_models.json
spectral-anomaly score-iforest artifacts/if/splits/calibration.parquet artifacts/if/window_iforest.pkl artifacts/if/calibration_scores.parquet
spectral-anomaly score-iforest artifacts/if/splits/evaluation.parquet artifacts/if/window_iforest.pkl artifacts/if/evaluation_scores.parquet
spectral-anomaly spot artifacts/if/calibration_scores.parquet artifacts/if/evaluation_scores.parquet artifacts/if/decisions.parquet artifacts/if/spot_state.pkl --config isolation_forest/configs/spot.json
spectral-anomaly plot-spot data/pmu.csv artifacts/if/decisions.parquet artifacts/if/decisions.html --index-col DATE_MESURE --value-col VIGY_1:VA_V1:MAG --source-id VIGY_1 --channel-id V1.MAG --sep ";"
```

Les partitions sont chronologiques (60/20/20 dans l'exemple). Les fenêtres de la
partition précédente qui recouvrent la suivante sont purgées. Une table marquée
`train` par le manifeste est utilisée entièrement pour l'entraînement, sans
nouveau découpage. Si l'on donne la table complète à `train-iforest`, seule sa
fraction train est utilisée.

Le modèle sauvegarde la forêt **et** le préprocesseur ajusté, l'ordre des
caractéristiques et la signature de leur construction. La commande de scoring
relit ces éléments ; elle n'ajuste rien aux nouvelles données.

Ouvrir `artifacts/if/decisions.html` dans un navigateur. Plotly est inclus dans le
fichier. Le signal brut figure en haut, les scores et le seuil en bas. Les zones
rouges couvrent les fenêtres détectées. Une zone non colorée peut aussi correspondre
à une période sans fenêtre évaluée. Le bouton de téléchargement du graphique est
configuré pour exporter en SVG.

## Nouveau signal avec une forêt déjà entraînée

Copier le modèle existant, par exemple sous `artifacts/if/window_iforest.pkl`, ainsi
que ses scores de calibration et leurs manifestes. Utiliser **les configurations
exactes ayant produit ses caractéristiques** pour préparer le nouveau CSV :

```bash
spectral-anomaly prepare-windows data/new_signal.csv artifacts/if/new_windows.parquet --config isolation_forest/configs/window_preprocessing.json --index-col DATE_MESURE --value-col VIGY_1:VA_V1:MAG --source-id VIGY_1 --channel-id V1.MAG --sep ";"
spectral-anomaly extract-windows artifacts/if/new_windows.parquet artifacts/if/new_features.parquet --config isolation_forest/configs/window_features.json
spectral-anomaly score-iforest artifacts/if/new_features.parquet artifacts/if/window_iforest.pkl artifacts/if/new_scores.parquet --batch-size 65536
spectral-anomaly spot artifacts/if/calibration_scores.parquet artifacts/if/new_scores.parquet artifacts/if/new_decisions.parquet artifacts/if/new_spot_state.pkl --config isolation_forest/configs/spot.json
spectral-anomaly plot-spot data/new_signal.csv artifacts/if/new_decisions.parquet artifacts/if/new_decisions.html --index-col DATE_MESURE --value-col VIGY_1:VA_V1:MAG --source-id VIGY_1 --channel-id V1.MAG --sep ";"
```

Si les scores de calibration manquent mais que les caractéristiques de calibration
existent, exécuter `score-iforest` sur celles-ci avec le modèle existant. Le modèle
ne contient pas à lui seul les observations nécessaires à la calibration.

Un changement de normalisation, taille de fenêtre, bandes, `top_changes`, qualité
ou recette d'extraction change le contrat. Reprendre les bonnes configurations au
lieu de supprimer le contrôle de compatibilité. Le graphique doit utiliser le CSV
du **même intervalle** que les décisions.

## Classer les fenêtres par score IF

Le fichier [rank_windows.py](rank_windows.py) lit les scores déjà calculés et
écrit un classement **du score le plus élevé au plus faible**, sans réentraînement.
Il utilise le Parquet produit par `score-iforest` et son
`.parquet.manifest.json`, conservé à côté du fichier.

Depuis la racine `det_anom`, après installation du paquet :

```bash
python isolation_forest/rank_windows.py artifacts/if/new_scores.parquet artifacts/if/ranked_windows.csv
```

Ouvrir **`artifacts/if/ranked_windows.csv`** pour consulter le classement. La
colonne `rank` commence à 1. Toutes les colonnes d'origine restent présentes,
notamment `score`, `source_id`, `channel_id`, `window_id`, `window_start` et
`window_end`. Les identifiants et les dates restent associés à leur score.
Les fenêtres de même score gardent leur ordre dans le fichier d'entrée.

Pour ne conserver que les 20 premiers rangs :

```bash
python isolation_forest/rank_windows.py artifacts/if/new_scores.parquet artifacts/if/top_20_windows.csv --top-n 20
```

`--top-n` doit être strictement positif ; sans cette option, toutes les fenêtres
sont exportées. S'il dépasse le nombre de fenêtres, toutes sont conservées.
Un fichier de scores vide produit un CSV avec les en-têtes seulement.
Le dossier de sortie est créé si nécessaire ; un CSV existant au même chemin est
remplacé. Les scores d'entrée restent inchangés. Les identifiants en double,
les scores non numériques/NaN/infinis et une convention de score incompatible
provoquent une erreur explicite.

Le score est **`-IsolationForest.score_samples`** : plus il est élevé, plus la
fenêtre est atypique selon la forêt. Ce n'est pas une probabilité. Le rang est
global sur le fichier fourni ; utiliser des scores issus du même modèle.
Le CSV classe les références des fenêtres et leurs scores ; pour tracer le signal
avec les décisions, voir `plot-spot` ci-dessus.

## Classement, corrélations et valeurs des caractéristiques

```bash
spectral-anomaly analyze-scores artifacts/if/new_features.parquet artifacts/if/new_scores.parquet artifacts/if/analysis --model artifacts/if/window_iforest.pkl --top-n 20
```

Cette analyse ne réentraîne pas le modèle. Elle produit :

| Sortie | Contenu |
|---|---|
| `ranked_windows.csv` | Toutes les fenêtres scorées, par score décroissant, avec identifiants et valeurs des caractéristiques |
| `top_windows.csv` | Les N premières lignes |
| `summary.json` | Nombre de fenêtres, statistiques des scores, caractéristiques utilisées |
| `scores.html` | Histogramme des scores |
| `correlation_pearson.csv` / `.html` | Corrélations linéaires entre caractéristiques |
| `correlation_spearman.csv` / `.html` | Corrélations entre rangs |

Avec `--model`, les corrélations portent sur les caractéristiques du modèle ; sans
cette option, elles portent sur celles du manifeste d'extraction. Elles sont
calculées sur **toutes les fenêtres scorées**, pas seulement le top N. Les
caractéristiques constantes produisent des corrélations indéfinies (NaN).

## Explications SHAP

```bash
python -m pip install -e "./isolation_forest[shap]"
spectral-anomaly-shap artifacts/if/window_iforest.pkl artifacts/if/new_features.parquet artifacts/if/shap --train artifacts/if/splits/train.parquet --top-n 20 --background-size 100 --permutations 20 --seed 42
```

`--train` est la table réellement utilisée à l'entraînement, avec son manifeste.
Pour une seule fenêtre, remplacer `--top-n 20` par `--window-id "IDENTIFIANT"` ou
`--date "2026-09-16T09:47:10Z"` (début exact). Les observations à expliquer sont les
**caractéristiques**, pas le fichier des scores.

Les sorties comprennent `global_importance.svg`, `beeswarm.svg`, `waterfall.svg`,
les valeurs SHAP en Parquet, le fond réellement utilisé et `analysis.json`.
L'additivité est vérifiée contre le score exact du pipeline sauvegardé.
Voir [le guide SHAP](docs/shap_window_iforest.md) pour toutes les options,
l'interprétation et les limites dues aux variables corrélées.

## Artefacts à conserver

| Artefact | Fichiers associés |
|---|---|
| Fenêtres | `windows.parquet`, `windows.parquet.manifest.json` |
| Audit du prétraitement | `windows.parquet.quality.parquet` et son manifeste |
| Caractéristiques / partitions / scores / décisions | Chaque Parquet et son `.parquet.manifest.json` |
| Modèle | `.pkl`, configurations d'origine, versions des dépendances |
| Seuil POT | `spot_state.pkl` et décisions ; la commande actuelle recalibre à partir du fichier de calibration |

`--resume` est disponible pour `score-iforest` et `spot` : il fusionne le Parquet
existant par identifiants après vérification des manifestes. Il ne reprend pas un
entraînement, ne recharge pas automatiquement un état SPOT et ne rend pas cette
chaîne batch équivalente à un détecteur de flux.

## Démonstration complète

Les constantes suivantes sont propres au **signal synthétique** : centre 100,
échelle 2, cadence 10 Hz. Elles ne constituent pas un réglage de données réelles.

```bash
python isolation_forest/examples/make_demo.py --output artifacts/if_demo
spectral-anomaly prepare-windows artifacts/if_demo/signal.csv artifacts/if_demo/windows.parquet --config artifacts/if_demo/window_preprocessing.json --source-id DEMO --channel-id V1.MAG
spectral-anomaly extract-windows artifacts/if_demo/windows.parquet artifacts/if_demo/features.parquet --config artifacts/if_demo/window_features.json
spectral-anomaly split-windows artifacts/if_demo/features.parquet artifacts/if_demo/splits --config artifacts/if_demo/window_models.json
spectral-anomaly train-iforest artifacts/if_demo/splits/train.parquet artifacts/if_demo/model.pkl --config artifacts/if_demo/window_models.json
spectral-anomaly score-iforest artifacts/if_demo/splits/calibration.parquet artifacts/if_demo/model.pkl artifacts/if_demo/calibration_scores.parquet
spectral-anomaly score-iforest artifacts/if_demo/splits/evaluation.parquet artifacts/if_demo/model.pkl artifacts/if_demo/evaluation_scores.parquet
spectral-anomaly spot artifacts/if_demo/calibration_scores.parquet artifacts/if_demo/evaluation_scores.parquet artifacts/if_demo/decisions.parquet artifacts/if_demo/spot.pkl --config artifacts/if_demo/spot.json
spectral-anomaly plot-spot artifacts/if_demo/signal.csv artifacts/if_demo/decisions.parquet artifacts/if_demo/decisions.html --source-id DEMO --channel-id V1.MAG
python isolation_forest/rank_windows.py artifacts/if_demo/evaluation_scores.parquet artifacts/if_demo/ranked_windows.csv
spectral-anomaly analyze-scores artifacts/if_demo/features.parquet artifacts/if_demo/evaluation_scores.parquet artifacts/if_demo/analysis --model artifacts/if_demo/model.pkl
```

## Aide et dépannage

Toutes les options et valeurs par défaut figurent dans
[CLI_REFERENCE.md](docs/CLI_REFERENCE.md), généré depuis les parseurs conservés.
`spectral-anomaly --help` et `spectral-anomaly COMMANDE --help` donnent la même aide.
Pour les logs détaillés : placer `--log-level DEBUG` **avant** la sous-commande.
`python -m spectral_anomaly ...` est un appel équivalent.

| Erreur / situation | Action |
|---|---|
| `finite mu and positive sigma` | Renseigner les constantes du canal |
| `not exactly aligned ... grid` | Vérifier cadence et horodatages du CSV |
| Aucune fenêtre acceptée | Lire l'audit qualité ; vérifier longueur, flags et lacunes |
| `band narrower ... FFT spacing` | Adapter bandes ou taille de fenêtre ; le zero-padding ne résout pas cette contrainte |
| `incompatible window_feature_signature` | Reprendre la recette exacte du modèle |
| `not enough excesses ... calibration` | Fournir plus de calibration représentative ou choisir explicitement d'autres paramètres ; le code ne les modifie pas |
| Aucune intersection temporelle au tracé | Scorer le CSV à tracer, puis utiliser ses propres décisions |
| Avertissement de version scikit-learn au chargement | Recréer l'environnement ayant produit le modèle |

Le backend CPU est le chemin validé. L'option GPU est conservée dans les modèles,
mais nécessite un environnement CUDA/cuML compatible et une validation sur cette
machine. Les caractéristiques sont calculées avec NumPy/SciPy sur CPU.
