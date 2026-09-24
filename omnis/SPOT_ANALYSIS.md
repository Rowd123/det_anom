# Analyse SPOT d'un checkpoint OmniAnomaly

Depuis la racine du dépôt, utiliser la configuration correspondant **exactement**
au modèle entraîné (architecture, colonnes, exclusions, normalisation et découpage) :

```bash
python omnis/analyze_spot.py \
  --config omnis/config.yaml \
  --checkpoint omnis/model/TON_MODELE.pt \
  --output artifacts/omni_spot \
  --top 50 --n-z 16 --batch-size 32 \
  --q 0.001 --level 0.98
```

Remplacer `TON_MODELE.pt` par le checkpoint réel. `--top 0` exporte toutes les
fenêtres ; cela peut générer beaucoup de SVG/NPZ. Toutes les fenêtres valides du
test sélectionnées par le pas sont scorées et incluses dans le CSV, quelle que soit la valeur de `--top`.
`--device cpu` force le CPU. `--seed 42` fixe la graine ; conserver aussi le même
batch-size et n-z pour reproduire les tirages Monte-Carlo.

Le chemin des données `data_dir` est relatif au fichier YAML, comme dans la
visualisation existante. Les chemins CLI checkpoint et output sont relatifs au
répertoire courant. Les champs d'entraînement du YAML (`run_mode`, `max_epoch`,
etc.) ne déclenchent jamais d'entraînement dans cette commande.

## Score et SPOT

La calibration utilise les fenêtres valides de `train` sélectionnées par le pas ; le flux analysé est
`test`, dans l'ordre chronologique. Choisir un train représentatif du régime
normal. Les masques d'exclusion et la représentation physique du dépôt sont
réutilisés. Le normaliseur est restauré depuis le checkpoint sans réajustement.

Le modèle est appelé avec `last_point_only=False`. Le score envoyé à SPOT
est calculé sur **toute la fenêtre** :

    score = -mean_time mean_z sum_features log p(x_t | z)

La somme porte sur les caractéristiques, la moyenne sur les échantillons latents
et sur tous les instants de la fenêtre. Si `get_score_on_dim` est activé,
le script effectue la somme des caractéristiques ; sinon le modèle la fournit.
Ce n'est pas une probabilité d'anomalie. Une perturbation intérieure contribue au
score même si le dernier point est normal ; un événement très bref peut toutefois
être atténué par la moyenne temporelle.

La reconstruction complète représente la moyenne des moyennes du décodeur pour
les mêmes tirages latents que ceux utilisés pour ce score. La calibration train,
l'analyse test et le classement utilisent tous cette définition. Les poids du
modèle et la normalisation sauvegardée restent inchangés. Les autres scripts
conservent leur convention de score existante.

Relancer la commande pour recalculer les scores et la calibration SPOT : les
anciens scores/seuils au dernier point ne sont pas réutilisables. Utiliser un
nouveau répertoire `--output` pour séparer les anciens et nouveaux exports.
`analysis.json` indique `score_aggregation: window_mean`.

SPOT fournit un seuil dynamique et une décision. Il n'existe pas ici de score
probabiliste SPOT distinct. `score` désigne le score OmniAnomaly fourni à SPOT ;
`excess = score - threshold` indique le dépassement du seuil, dans les unités du
score. Le classement demandé est par `score` décroissant, avec ordre temporel en
cas d'égalité. Le tri intervient **après** l'analyse séquentielle de SPOT.

`--level` est un quantile de queue haute (0.98 par défaut), à distinguer du champ
historique `level: 0.01` utilisé en queue basse dans le YAML d'entraînement.
Il faut `0 < q < 1-level`. Une calibration avec moins de cinq excès, des excès
constants ou des valeurs non finies est refusée explicitement.

## Sorties

- `index.html` : liens vers les fenêtres exportées, classées par score décroissant.
- `ranked_windows.csv` : toutes les fenêtres avec rang, indices, score, seuil
  réellement utilisé pour la décision, dépassement, décision et chemins d'export.
- `rank-XXXXXX_window-Y.svg` : toutes les grandeurs physiques reconstructibles,
  signal observé/reconstruit, score, seuil SPOT et décision.
- `rank-XXXXXX_window-Y.npz` : tableaux physiques observés/reconstruits, noms des
  grandeurs, indices et résultat SPOT de la fenêtre.
- `chronological_scores.npz` : scores et décisions dans l'ordre du flux original.
- `calibration_scores.npz` et `analysis.json` : calibration et paramètres du run.

Les indices `window_start/end` sont relatifs à test ; `source_start/end` sont les
indices de lignes des données complètes (base zéro, bornes inclusives), après le
chargement habituel du dépôt. Ce ne sont pas des timestamps. Les fenêtres qui
contiennent des points exclus sont absentes, sans renumérotation des indices.
Les features exclues du modèle ne sont pas présentées comme reconstruites.

Les reconstructions conservées en mémoire sont limitées au top demandé ; le
calcul des scores garde le fonctionnement par mini-batch. Aucun checkpoint n'est
modifié. Dépendances : celles du projet (PyTorch, pyro-ppl, NumPy, pandas, SciPy,
PyYAML, tqdm, matplotlib). Les exports SVG ne nécessitent pas Kaleido.


## Fenêtres espacées pour accélérer l'analyse

Par défaut, le pas vaut maintenant `window_length` : avec 100 points par fenêtre,
les fenêtres sont 0–99, 100–199, 200–299, etc. Elles ne se recouvrent pas.
`--stride 50` avance de 50 échantillons ; `--stride 1` retrouve le pas
historique, avec le nouveau score sur toute la fenêtre. Le même pas est appliqué à train (calibration) et test, **avant**
l'inférence : les fenêtres intermédiaires ne sont pas calculées.

Chaque partie commence sa propre grille à son premier point. Une fenêtre avec un
point exclu est ignorée sans décaler cette grille. Les derniers points qui ne
forment pas une fenêtre complète sont ignorés. Le pas effectif est enregistré
dans `analysis.json`. L'entraînement et les autres scripts conservent leur pas
historique de 1 ; la longueur des fenêtres du modèle reste inchangée.

Avec 100 points par fenêtre et un pas de 100, chaque décision porte sur les
100 points de la fenêtre correspondante. Le modèle reconstruit la fenêtre entière.
SPOT reçoit moins de scores de calibration : si le nombre d'excès est insuffisant,
augmenter la durée de calibration ou réduire `--stride`. Le script refuse de
fabriquer un seuil avec une calibration insuffisante.

