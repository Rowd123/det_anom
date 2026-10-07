# Configuration de la méthode par fenêtres

Tous les chemins sont relatifs à la racine du projet.

## Configuration du prétraitement

`isolation_forest/configs/window_preprocessing.json` : fréquence, période, taille/recouvrement,
qualité, exclusions et normalisation. Les valeurs `mu` et `sigma` sont
intentionnellement `null` dans l'exemple : **renseigner vos constantes**, sinon la
commande échoue. Elles ne sont jamais estimées. Exemple de structure :

```json
"normalization": {
  "V1.MAG": {"mu": null, "sigma": null},
  "FREQ": {"mu": null, "sigma": null}
}
```

Les canaux correspondent à `--channel-id`; exécuter le prétraitement séparément
pour chaque série. Le calcul est `(x - mu) / sigma`, avec sigma strictement positif.
Le code conserve `prepare_analysis_windows` de la source : mêmes
règles de tri, doublons, grille temporelle, qualité, interpolation et exclusions.
Il utilise son signal, sans soustraction supplémentaire de la moyenne locale.
L'audit est sauvegardé dans `<windows>.quality.parquet`.

L'exemple propose 50 Hz et 3000 échantillons (60 s), avec un pas de 30 s : ce sont
des paramètres d'exemple, à adapter aux données. Les fenêtres incomplètes en fin
de série ne sont pas utilisées, conformément au prétraitement existant.

## Configuration de l'extraction

`isolation_forest/configs/window_features.json` contrôle tous les descripteurs :

| Paramètre | Effet |
|---|---|
| `statistics` | Sous-ensemble de minimum, maximum, mean, variance, median, iqr, skewness, kurtosis |
| `variance_ddof` | Correction de variance, par défaut 0 |
| `moment_bias` | Option SciPy des estimateurs de skewness/kurtosis ; kurtosis est l'excès de Fisher |
| `variance_epsilon` | Seuil de variance pour considérer une fenêtre constante |
| `undefined_policy` | `zero` ou `error` pour moments/autocorrélations indéfinis |
| `differences` | Liste facultative : mean_absolute, maximum_absolute, variance ; vide = désactivé |
| `spectral.enabled` | Active les puissances par bandes |
| `spectral.bands_hz` | Bornes non uniformes, ordonnées et non chevauchantes ; couverture partielle autorisée |
| `spectral.window`, `detrend`, `nfft` | Paramètres du périodogramme ; `detrend: false` ne retire pas de moyenne |
| `spectral.top_changes` | Nombre des variations de bandes de plus grande valeur absolue ; 0 désactive |
| `temporal.lags_seconds` | Retards exacts en multiples de la période d'échantillonnage |
| `temporal.changes` | Ajoute les écarts d'autocorrélation par rapport à l'historique |
| `history.size`, `min_windows` | Taille maximale et minimum requis de l'historique |
| `history.warmup` | `drop`, `zero` ou `error` avant d'avoir assez d'historique |
| `history.reset_after_gap_seconds` | Écart maximal entre débuts successifs ; null = 1,5 fois le pas des fenêtres |
| `log_every` | Fréquence des messages de progression |

Les noms sont préfixés `wf_`. `wf_band_0_power` correspond à la première bande,
`wf_acf_0` au premier retard. Les variations utilisent le suffixe `_change`.
`wf_spectral_change_0` est la plus forte variation **signée** de puissance ;
`wf_spectral_change_0_frequency` est le centre de sa bande, pas la fréquence d'un
pic estimé. Les égalités sont départagées par l'ordre des bandes.

Le calcul spectral est un périodogramme unilatéral en densité, puis la somme des
bins sélectionnés multipliée par leur espacement. Les bandes sont semi-ouvertes
[bas, haut[, sauf la dernière borne si elle égale Nyquist. Elles doivent contenir
des bins et leur largeur doit atteindre au moins fs/N ; le zero-padding ne remplace
pas une fenêtre longue. Ces puissances sont celles du signal normalisé et non une
énergie physique en joules. Aucun estimateur de phase n'est utilisé.

L'autocorrélation au retard k est `sum(y[:-k]*y[k:]) / sum(y*y)` avec
`y = x - mean(x)`. Ce centrage définit la caractéristique, sans modifier les données
prétraitées. `wf_constant` indique une fenêtre où les moments normalisés sont
indéfinis et remplacés par zéro selon la politique choisie. Les différences sont
des différences finies non divisées par la période, pas des dérivées continues.

Les références historiques sont les médianes des fenêtres **strictement
précédentes**, séparément par source et canal. Elles sont réinitialisées après un
écart temporel trop grand. `wf_history_ready` indique la disponibilité de la référence.
Avec `warmup: drop`, les premières fenêtres alimentent la référence mais sont
absentes de la table finale. Une extraction se fait sur un historique complet :
ne pas traiter chaque fenêtre dans une invocation distincte.

Le prétraitement historique conserve son interpolation bornée, qui peut utiliser
un échantillon suivant pour remplir une petite lacune. La causalité annoncée ici
concerne la référence des caractéristiques, pas une nouvelle garantie de streaming
strict du prétraitement existant.


## Modèle, partitions et seuil

Dans `isolation_forest/configs/window_models.json` :

| Clé | Emploi |
|---|---|
| `preprocessing.atypicality_features` | Liste ordonnée et explicite des caractéristiques retenues |
| `preprocessing.atypicality_exclude` | Motifs glob à retirer, par exemple `wf_acf_*` ; un motif inconnu est refusé |
| `preprocessing.log1p` | Transformation facultative avant mise à l'échelle, réservée aux valeurs > -1 |
| `preprocessing.scaling` | `none`, `standard` (moyenne/écart-type du train), `robust` (médiane/1,4826 MAD du train) |
| `split.train_fraction`, `calibration_fraction`, `evaluation_fraction` | Fractions des fenêtres, non négatives, somme 1 |
| `split.strategy`, `time_column`, `purge_overlap` | Ordre chronologique, colonne temporelle et purge du recouvrement ; purge toujours activée pour les artefacts de fenêtres |
| `atypicality.backend` | `cpu`, `gpu` ou `auto` ; auto utilise cuML seulement si disponible avec CUDA |
| `atypicality.type` | `isolation_forest` |
| `atypicality.n_estimators` | Nombre d'arbres |
| `atypicality.max_samples` | Taille du sous-échantillon par arbre, dont `auto` |
| `atypicality.contamination` | Paramètre de la forêt ; ne transforme pas le score en probabilité et ne fixe pas le seuil POT |
| `atypicality.n_jobs` | Parallélisme du backend CPU |
| `reproducibility.random_seed` | Graine fournie à la forêt, sauf `random_state` explicitement fixé dans `atypicality` |
| `reproducibility.artifact_version`, `preprocessing_version` | Métadonnées conservées dans la configuration ; le format des manifestes a sa propre version |
| `spot` | Paramètres conservés dans la configuration modèle ; la CLI `spot` lit effectivement le fichier passé par `--config` |

La mise à l'échelle des **caractéristiques** est distincte de la normalisation fixe
du **signal**. La sélection de variables ne modifie pas le fichier extrait. Un
changement de sélection nécessite de réentraîner le modèle concerné, pas de
recalculer les caractéristiques déjà présentes.

Dans `isolation_forest/configs/spot.json`, `q` est un paramètre de queue extrême,
`initial_quantile` définit le seuil initial et `min_excesses` le nombre minimal
d'excès strictement positifs pour l'ajustement de Pareto généralisé. Fournir une
calibration normale suffisamment longue, finie et variable, avec
`0 < q < 1 - initial_quantile` et `0 < initial_quantile < 1`. Le nombre requis de
fenêtres dépend des égalités éventuelles au quantile. Le seuil final reste fixe.

## Qualité et exclusions

`quality_column: null` désactive les flags ; sinon fournir `valid_flags`, par
exemple `["NORMAL"]`. `min_valid_fraction` porte sur les échantillons réellement
observés, avant interpolation. Seules les lacunes internes bornées ne dépassant
pas `max_interpolation_gap` sont interpolées. Les fenêtres touchant un événement
explicitement exclu sont rejetées.

`exclusions.events` accepte des objets avec `event_id`, `source_id` facultatif,
et/ou `start` et `end` (bornes inclusives). Les IDs utilisent `event_id_column` ;
`unknown_id_policy` vaut `warning` ou `error`. Les dates d'exclusion doivent avoir
un fuseau compatible avec les dates UTC du CSV. Exemple :

```json
{"events": [{"source_id": "VIGY_1", "start": "2026-09-01T12:00:00Z", "end": "2026-09-01T12:10:00Z"}], "event_id_column": null, "unknown_id_policy": "error"}
```

## Contrat des données

Chaque `window_features` porte une signature de sa configuration complète de
prétraitement et d'extraction. Le modèle enregistre cette signature et refuse
les caractéristiques incompatibles. Les identifiants sont composés de la source,
du canal et du début de fenêtre. Garder des IDs stables entre extraction, scores,
calibration et décisions.
