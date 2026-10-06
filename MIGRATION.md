# Migration depuis les trois dépôts

## Sources retenues

| Nouvelle partie | Dépôt et branche source | Commit |
|---|---|---|
| `isolation_forest` | `Rowd123/spec_anom`, `feature/window-statistics` | `b95d4d9cf67fb9629fb0f054beadbcd97d52c6a4` |
| `rpad_detection` | `Rowd123/PMU-simulator`, `feature/rpad-event-detection` | `5b7319ce8b1b0e7c880dc5d4f24fdae2ca83367d` |
| `omnis` | `Rowd123/det_anom`, `main`, dossier `omnis` et tests associés | `9e2f4517f6d3db1564650edb5c26053d232a46b7` |

Le regroupement est publié dans `Rowd123/det_anom`. Les trois méthodes sont
autonomes, sans sous-modules Git ni imports vers les anciens répertoires.
Les commits ci-dessus identifient les versions source ; l’historique de `det_anom`
conserve son organisation précédente.

## Isolation Forest

Conservés : qualité/interpolation, exclusions, normalisation fixe, fenêtres,
descripteurs temporels et fréquentiels, sélection des caractéristiques, partitions
chronologiques avec purge, entraînement et scoring IF, calibration POT, Plotly et
SHAP. Retirés : SAM, segmentation d'images, MSST/STFT comme pipelines de segmentation,
HDBSCAN et leurs configurations/exemples. Les puissances spectrales calculées dans
chaque fenêtre restent présentes : elles font partie des caractéristiques IF.

Le nom Python `spectral_anomaly`, les classes `ModelPipeline`, `AtypicalityModel`
et `FeaturePreprocessor`, ainsi que les champs des artefacts sont conservés pour
les pickle existants. Les tables de fenêtres gardent `segment_id=0` : il s'agit
d'une clé historique de traçabilité, pas d'un masque de segmentation.

Pour une forêt déjà entraînée sur `wf_*` :

1. Copier le `.pkl`, les caractéristiques d'entraînement/calibration et leurs
   `.parquet.manifest.json`.
2. Conserver les versions de NumPy/scikit-learn utilisées pour créer le pickle.
3. Réutiliser les mêmes configurations de prétraitement et d'extraction pour les
   nouvelles données. Ne pas remplacer ses paramètres par les exemples fournis.
4. Appeler `score-iforest`, puis `spot` et/ou `plot-spot` ; aucun entraînement
   supplémentaire n'est requis.

Les anciens modèles basés sur des segments SAM ne font pas partie de ce périmètre.
Le nom de distribution `spectral-anomaly` est conservé : installer dans un
environnement propre évite que deux copies du même paquet Python soient utilisées.

L'ancien script d'analyse mélangeant IF, HDBSCAN et galeries de segments a été
remplacé par `analyze-scores` pour les tableaux/classements/corrélations. Le module
SHAP dédié aux fenêtres est conservé ; il utilise un fond issu du train.

## RPAD

Les modules `pmu_simulator.detection` restent importables. Le point d'entrée
principal est `rpad-detection` ; `pmu-sim` reste un alias pour ses **trois commandes
de détection**. Les commandes `simulate`, `animate`, les entrées `--simulation-a/b`
et les modules de génération/estimation PMU sont retirés.

L'entrée standard reste un CSV de deux angles synchronisés ou un flux NDJSON.
L'option additive `--time-format iso8601` accepte des dates avec fuseau et les
convertit en secondes Unix ; `seconds` reste le défaut. Les algorithmes de
déroulement, interpolation, critères, seuils et votes sont conservés.

Les états JSON RPAD ont la même structure. Leur réutilisation demande la même
configuration et le suffixe temporel encore non traité, comme auparavant.

## OmniAnomaly

Le contenu fonctionnel de `omnis` est conservé, sans les anciens dossiers CNN/LSTM
du dépôt. Les caches Python et poids étrangers à `omnis` ne sont pas copiés.
Les imports internes deviennent relatifs au paquet `omnis` pour éviter les
collisions entre des noms génériques comme `utils`, `model` ou `spot`.

Les scripts directs restent utilisables, et des points d'entrée sont ajoutés :

| Ancien appel | Appel installé |
|---|---|
| `python omnis/main.py --config ...` | `omni-run --config ...` |
| `python omnis/analyze_spot.py ...` | `omni-spot ...` |
| `python omnis/visualize_reconstruction.py ...` | `omni-reconstruct ...` |

Les tenseurs et clés de checkpoint, l'ordre des caractéristiques et les
normaliseurs restent inchangés. Le chargement vérifie toujours leur compatibilité.
Un argument `normalizer` inutilisé a été retiré de `SigmaEpochStatistics` pour
satisfaire un test déjà présent dans la source ; le normaliseur de `Trainer` est
bien conservé.

Deux erreurs d'exécution de la source ont également été corrigées : la restauration
du meilleur checkpoint en fin d'entraînement référençait une variable `config`
absente de `Trainer.fit` ; elle restaure maintenant l'état sauvegardé. Le parsing
des dates CSV accepte une précision fractionnaire variable et normalise les dates
en UTC, sans modifier les valeurs des signaux ni les masques de qualité.

**Chemins :** dans le YAML d'entraînement, `data_dir`, `save_dir`, `result_dir` et
`restore_checkpoint` sont désormais relatifs au fichier YAML. C'était déjà la
convention des données dans les scripts d'analyse. Un `--checkpoint` fourni en
ligne de commande est relatif au répertoire courant dans les trois commandes.
Les `--output` des outils d'analyse sont aussi relatifs au répertoire courant.
Les chemins absolus restent acceptés. Adapter les anciens YAML qui supposaient
le répertoire courant pour les chemins d'entraînement.

`omni-run` accepte maintenant `--run-mode scratch|resume|evaluate` et
`--checkpoint`. En mode `scratch`, un checkpoint en argument est refusé plutôt
que silencieusement ignoré.

## Données et résultats

Les configurations et petits générateurs de démonstration sont fournis. Les
mesures de terrain et les modèles de l'utilisateur ne sont pas intégrés au
dépôt regroupé. Leurs emplacements doivent être renseignés avant de lancer les commandes.
Les `.pkl` ne doivent être chargés que depuis une source de confiance.
