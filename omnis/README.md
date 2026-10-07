# OmniAnomaly pour les données PMU

Toutes les commandes partent de **`det_anom/`**.
Installation : `python -m pip install -e ./omnis`.
Les dépendances incluent PyTorch, `pyro-ppl`, NumPy, pandas, SciPy, PyYAML,
Matplotlib, Plotly et tqdm. Le chemin CPU est utilisable ; `device: cuda` demande
une installation PyTorch et une machine disposant de CUDA.

Trois commandes sont disponibles :

| Commande | Fonction |
|---|---|
| `omni-run` | Entraîner (`scratch`), reprendre (`resume`) ou évaluer (`evaluate`) |
| `omni-spot` | Charger un checkpoint, scorer des fenêtres complètes, appliquer SPOT et exporter les fenêtres classées |
| `omni-reconstruct` | Charger un checkpoint et tracer une reconstruction continue au dernier point |

Les scripts `python omnis/main.py`, `python omnis/analyze_spot.py` et
`python omnis/visualize_reconstruction.py` restent utilisables avec les mêmes
options. Les équivalents `python -m omnis.main`, `python -m omnis.analyze_spot` et
`python -m omnis.visualize_reconstruction` fonctionnent aussi.

## Données et chemins

Renseigner `data_dir` dans `omnis/config.yaml`. Le chemin d'exemple `../data/pmu.csv`
désigne `det_anom/data/pmu.csv`, car les chemins de données du YAML
sont relatifs au **fichier de configuration**.

Le chargeur PMU lit un CSV séparé par `;`, avec :

- une colonne temporelle `DATE_MESURE` ;
- les 14 grandeurs physiques énumérées par `data_column_order` : `DFDT`, `FREQ`,
  puis `ANG`/`MAG` pour `I0`, `I1`, `I2`, `V0`, `V1`, `V2` ;
- facultativement des colonnes dont le nom finit par `QF`. Chaque flag doit être
  `NORMAL` pour que la ligne soit valide.

Les noms configurés sont résolus vers les noms réels du CSV, éventuellement
préfixés par le poste. L'ordre est explicite et conservé. Les phases physiques
sont en degrés. La représentation du modèle conserve la magnitude et représente
la phase par sinus/cosinus ; `x_dim` est dérivé, pas choisi arbitrairement.

Les dates sont converties en UTC, y compris lorsque la précision des fractions de seconde varie. Les doublons temporels, lignes signalées, NaN et valeurs non finies sont exclus.
Les fenêtres contenant un point exclu sont écartées. Le chargeur conserve l'ordre
des lignes ; **fournir des mesures déjà triées et à cadence régulière**. Il
n'effectue pas la régularisation/interpolation du module Isolation Forest.
Dans le chargeur historique, un NaN dans une colonne du CSV peut exclure la ligne
avant la sélection des 14 grandeurs.

`max_train_size: 0.8` prend les premiers 80 % des lignes pour train ; le reste est
test. Un entier fixe un nombre de lignes. `train_start` choisit le premier indice,
`max_test_size` borne le test. Les mêmes réglages sont nécessaires pour retrouver
la même calibration et les mêmes indices dans les analyses.

| Chemin | Origine |
|---|---|
| YAML : `data_dir`, `save_dir`, `result_dir`, `restore_checkpoint` | Dossier du YAML |
| YAML de visualisation : `checkpoint` | Dossier du YAML |
| CLI : `--config`, `--checkpoint`, `--output` | Répertoire courant |
| YAML de visualisation : `output` | Répertoire courant |

Les chemins absolus sont acceptés. Avec `omnis/config.yaml`, `save_dir: model` et
`result_dir: result` produisent `omnis/model/` et `omnis/result/`.

## Entraîner un nouveau modèle

Adapter `omnis/config.yaml`, puis :

```bash
omni-run --config omnis/config.yaml --run-mode scratch
```

Paramètres essentiels :

| Groupe | Paramètres |
|---|---|
| Représentation | `data_column_order`, `excluded_features`, `exclude_zero_sequence_phase` |
| Normalisation | `normalization_mode`, `nominal_normalization` |
| Architecture | `window_length`, `z_dim`, `rnn_num_hidden`, `dense_dim`, `use_connected_z_q`, `use_connected_z_p` |
| Décodeur | `decoder_sigma_mode`, `decoder_fixed_sigma`, `std_epsilon` |
| Apprentissage | `max_epoch`, `max_step`, `batch_size`, `train_n_z`, `initial_lr`, `l2_reg`, `gradient_clip_norm` |
| Validation / arrêt | `valid_portion`, `valid_batch_size`, `valid_n_z`, `early_stop`, `early_stopping_validation_patience` |
| Décroissance du taux | `lr_anneal_epoch_freq`, `lr_anneal_factor` |
| Journalisation | `train_log_step_freq`, `deep_training_diagnostics` |
| Filtrage robuste optionnel | `anomaly_filter_enabled`, `anomaly_filter_warmup_epochs`, `anomaly_filter_k`, `anomaly_filter_history_size`, `anomaly_filter_min_history` |
| Inférence historique | `test_batch_size`, `test_n_z`, `get_score_on_dim`, `save_z` |

Trois normalisations sont conservées : `train_minmax` ajustée sur le train,
`nominal` utilisant les constantes configurées, `window_local` appliquée fenêtre
par fenêtre avec son contexte de reconstruction. Le normaliseur est sauvegardé
dans le checkpoint ; il ne doit pas être réajusté à l'inférence.

La configuration exhaustive, les valeurs par défaut et les champs historiques
figurent dans [CONFIGURATION.md](docs/CONFIGURATION.md). Les normalizing flows
ne sont pas implémentés dans cette version ; `posterior_flow_type` reste `null`.
Ne pas interpréter un champ conservé dans le YAML comme la prise en charge d'une
nouvelle architecture : le code emploie les réseaux récurrents existants.

Le programme affiche le chemin du meilleur checkpoint, dont le nom dépend des
paramètres et de l'empreinte des caractéristiques. **Recopier ce nom exact** ;
ne pas supposer qu'il s'appelle simplement `best_omni_anomaly.pt`.

## Reprise et évaluation sans entraînement

Remplacer `MODELE.pt` par le checkpoint réel :

```bash
omni-run --config omnis/config.yaml --run-mode resume --checkpoint omnis/model/MODELE.pt
omni-run --config omnis/config.yaml --run-mode evaluate --checkpoint omnis/model/MODELE.pt
```

`resume` restaure le modèle, le normaliseur et l'état d'entraînement sauvegardé.
L'entraînement reprend selon les paramètres du YAML. `evaluate` saute
`trainer.fit` et calcule les scores avec les poids restaurés. Il écrit néanmoins
les résultats dans `result_dir` : choisir un autre dossier pour conserver un run
précédent. Donner `--checkpoint` avec `scratch` provoque une erreur explicite.

Ces options peuvent aussi être fixées par `run_mode` et `restore_checkpoint`
dans le YAML ; la CLI est prioritaire. Le fichier de données, les colonnes,
exclusions, architecture et normalisation doivent correspondre au checkpoint.

## Analyse SPOT de fenêtres complètes

Cette commande charge un modèle existant **sans entraînement**, quels que soient
les paramètres `run_mode` ou `max_epoch` du YAML :

```bash
omni-spot --config omnis/config.yaml --checkpoint omnis/model/MODELE.pt --output artifacts/omni_spot --top 50 --batch-size 32 --n-z 16 --q 0.001 --level 0.98 --seed 42
```

La calibration utilise train, l'analyse utilise test dans l'ordre chronologique.
Le score est :

```text
score = - moyenne_temps moyenne_tirages_latents somme_caractéristiques log p(x_t | z)
```

Le score couvre **toute la fenêtre** ; il ne s'agit pas d'une probabilité d'anomalie.
SPOT donne un seuil dynamique et une décision. `excess = score - threshold`
mesure le dépassement. Le classement par score décroissant est effectué après
l'analyse séquentielle, sans réordonner l'entrée de SPOT.

Le pas par défaut est `window_length`, donc sans recouvrement. `--stride 50`
avance de 50 échantillons ; `--stride 1` conserve toutes les fenêtres glissantes.
Le même pas s'applique à la calibration et au test, **avant l'inférence**.
`--top 0` exporte toutes les reconstructions ; sinon toutes les fenêtres sont
scorées, mais seules les N premières reconstructions sont conservées pour la galerie.

La calibration doit être suffisamment longue et variable pour produire au moins
cinq excès au quantile `--level`. Il faut `0 < q < 1 - level`. Le `--level 0.98`
de cette analyse en queue haute diffère du champ historique `level: 0.01` employé
dans l'évaluation en queue basse de `omni-run`.

| Sortie sous `--output` | Contenu |
|---|---|
| `index.html` | Galerie classée, un lien par fenêtre exportée |
| `ranked_windows.csv` | Toutes les fenêtres : rang, indices, score, seuil au moment de la décision, dépassement, alarme et fichiers |
| `rank-XXXXXX_window-Y.svg` | Signal observé/reconstruit de la fenêtre, grandeurs physiques et décision |
| `rank-XXXXXX_window-Y.npz` | Tableaux de la fenêtre et métadonnées |
| `chronological_scores.npz` | Scores, indices, seuils et alarmes dans l'ordre temporel |
| `calibration_scores.npz` | Scores et indices de calibration |
| `analysis.json` | Configuration, checkpoint, graines, pas, conventions et compteurs |

Les indices `window_start/end` sont relatifs au test ; `source_start/end` sont les
indices des lignes des données complètes, base zéro et bornes inclusives. Ce ne
sont pas des timestamps. Les fenêtres exclues ne sont pas renumérotées.
Le pas change le nombre de scores et la calibration : recalculer les seuils
lorsqu'il change. Les modèles et normaliseurs restent inchangés.

## Reconstruction continue

Adapter `omnis/visualize_config.yaml` aux paramètres du checkpoint, puis :

```bash
omni-reconstruct --config omnis/visualize_config.yaml --checkpoint omnis/model/MODELE.pt --split test --start 99 --points 1000 --batch-size 32 --n-z 16 --backend plotly --output artifacts/omni/reconstruction.html
omni-reconstruct --config omnis/visualize_config.yaml --checkpoint omnis/model/MODELE.pt --split test --start 99 --points 1000 --backend matplotlib --output artifacts/omni/reconstruction.png
```

`--start` est l'indice du premier point final demandé dans la partition choisie ;
prévoir au moins `window_length - 1` points de contexte. Cette visualisation
reconstruit les **derniers points** des fenêtres glissantes et garde son
comportement historique. Pour visualiser des fenêtres entières classées par score,
utiliser `omni-spot` et ses SVG. Le backend `plotly` exige `.html`, `matplotlib`
exige `.png`. Les variables exclues du modèle ne sont pas présentées comme
reconstruites.

## Résultats d'entraînement et limites des métriques

`omni-run` sauvegarde dans `result_dir` la configuration effective, les scores
`train_score.pkl` / `test_score.pkl` et `metrics.json`, selon les noms activés dans
le YAML. Les checkpoints sont dans `save_dir` : le meilleur porte un nom paramétré, et `omni_anomaly.pt` est la sauvegarde de fin de run (le meilleur état est restauré en fin de fitting).
Les scores historiques de `Predictor` portent sur le dernier point et suivent
la convention de log-vraisemblance utilisée par `omni-run`. Ils ne remplacent pas
les scores négatifs de fenêtre complète de `omni-spot`.

Le chargeur CSV PMU actuel renvoie `test_label=None`. Il ne fournit donc pas
automatiquement précision, rappel, F1 ou ROC-AUC pour ce CSV. Les fonctions
d'évaluation supervisée sont conservées dans `eval.py`, mais nécessitent des
étiquettes réellement alignées aux scores. Les métriques de perte ou de durée
d'exécution ne constituent pas ces métriques de détection.

## Dépannage et tests

| Message / situation | À vérifier |
|---|---|
| Fichier introuvable | Résolution des chemins depuis le YAML ou la CLI |
| Colonne absente/ambiguë | `data_column_order`, préfixes réels et choix d'un seul poste |
| Ordre/noms de features incompatibles | Colonnes, exclusions et recette ayant produit le checkpoint |
| Paramètres nominaux `center`/`scale` différents | Restaurer les constantes du checkpoint, sans désactiver la vérification |
| Dimension ou `state_dict` incompatible | Architecture et représentation exactes du modèle |
| Pas assez d'excès SPOT | Augmenter la durée de calibration, diminuer le pas, ou choisir explicitement un autre quantile |
| Calcul trop lent | Régler le pas de `omni-spot`, puis batch-size/n-z selon les ressources ; ne pas modifier la longueur attendue par le modèle |

```bash
python -m pytest omnis/tests
omni-run --help
omni-spot --help
omni-reconstruct --help
```

[CLI_REFERENCE.md](docs/CLI_REFERENCE.md) liste toutes les options.
[SPOT_ANALYSIS.md](SPOT_ANALYSIS.md) détaille la convention et les exports SPOT.
Les tests portent sur de petits cas synthétiques ; un checkpoint de production
et ses données doivent être validés dans leur environnement d'origine.
