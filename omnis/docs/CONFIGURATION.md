# Référence des configurations OmniAnomaly

Toutes les valeurs ci-dessous sont celles des exemples fournis, pas une recommandation de réglage pour un réseau. Conserver les paramètres ayant servi à créer un checkpoint.

## `config.yaml`

| Clé | Valeur fournie | Usage |
|---|---|---|
| `dataset` | `"PMU"` | Conserver PMU ; le chargeur actuel lit le CSV PMU et contrôle les 14 grandeurs physiques. |
| `data_dir` | `"../data/pmu.csv"` | CSV séparé par ;, chemin relatif au YAML. |
| `data_column_order` | `["DFDT", "FREQ", "I0.ANG", "I0.MAG", "I1.ANG", "I1.MAG", "I2.ANG", "I2.MAG", "V0.ANG", "V0.MAG", "V1.ANG", "V1.MAG", "V2.ANG", "V2.MAG"]` | Noms ou suffixes ordonnés des 14 grandeurs physiques ; pas les colonnes sinus/cosinus du modèle. |
| `x_dim` | `null` | Dimension dérivée à partir de la représentation ; ne pas la fixer manuellement. |
| `exclude_zero_sequence_phase` | `false` | Exclut les phases des séquences homopolaires I0/V0, en conservant leurs magnitudes. |
| `excluded_features` | `[]` | Liste de grandeurs configurées à retirer de la représentation ; le chargeur attend toujours ses colonnes physiques. |
| `normalization_mode` | `"train_minmax"` | train_minmax, nominal ou window_local (min–max par fenêtre). |
| `nominal_normalization` | `{"DFDT": {"center": 0.0, "scale": 1.0}, "FREQ": {"center": 50.0, "scale": 0.5}, "I0.MAG": {"center": 0.0, "scale": 1.0}, "I1.MAG": {"center": 0.0, "scale": 1.0}, "I2.MAG": {"center": 0.0, "scale": 1.0}, "V0.MAG": {"center": 0.0, "scale": 240000.0}, "V1.MAG": {"center": 0.0, "scale": 240000.0}, "V2.MAG": {"center": 0.0, "scale": 240000.0}}` | Constantes center/scale par suffixe de grandeur. Appliquées aux grandeurs directes et magnitudes ; sinus/cosinus ont centre 0, échelle 1. |
| `use_connected_z_q` | `true` | Active le posterior latent récurrent conditionné par le latent précédent. |
| `use_connected_z_p` | `true` | Active le prior latent connecté (marche aléatoire gaussienne). |
| `z_dim` | `3` | Dimension latente. |
| `rnn_cell` | `"GRU"` | Le modèle actuel impose GRU. Ce champ historique sert au nom du checkpoint ; le conserver à GRU. |
| `rnn_num_hidden` | `500` | Taille cachée des réseaux GRU. |
| `window_length` | `100` | Nombre de points par fenêtre ; doit correspondre au checkpoint. |
| `dense_dim` | `500` | Largeur des couches denses. |
| `posterior_flow_type` | `null` | Champ historique non implémenté ; conserver null. |
| `nf_layers` | `20` | Champ historique sans effet tant que les flows ne sont pas implémentés. |
| `std_epsilon` | `0.005` | Plancher de sigma pour les distributions à écart-type appris. |
| `decoder_sigma_mode` | `"fixed"` | fixed ou learned. |
| `decoder_fixed_sigma` | `0.05` | Écart-type du décodeur en mode fixed. |
| `run_mode` | `"scratch"` | scratch, resume ou evaluate ; surcharge possible avec --run-mode. |
| `restore_checkpoint` | `null` | Checkpoint pour resume/evaluate, relatif au YAML ; --checkpoint CLI a priorité et est relatif au répertoire courant. |
| `max_epoch` | `50` | Numéro maximal d’époque ; après reprise, il ne s’agit pas d’un nombre d’époques supplémentaires. |
| `max_step` | `null` | Plafond de mises à jour globales, ou null. |
| `train_start` | `0` | Indice de la première ligne de la partie train. |
| `max_train_size` | `0.8` | Fraction si flottant <= 1, sinon nombre de lignes ; null prend tout le reste et laisse le test vide. |
| `batch_size` | `50` | Fenêtres par lot d’entraînement ; les lots incomplets sont écartés. |
| `valid_batch_size` | `50` | Fenêtres par lot de validation. |
| `train_log_step_freq` | `100` | Nombre de mises à jour entre deux journaux de progression. |
| `train_n_z` | `1` | Tirages latents par fenêtre pour l’entraînement. |
| `valid_n_z` | `8` | Tirages latents pour la validation. |
| `deep_training_diagnostics` | `false` | Active les diagnostics supplémentaires de sigma, NLL et gradients ; coût de calcul supplémentaire. |
| `anomaly_filter_enabled` | `false` | Filtre robuste des pertes de fenêtres pendant l’entraînement. |
| `anomaly_filter_warmup_epochs` | `2` | Époques initiales sans filtrage robuste. |
| `anomaly_filter_k` | `5.0` | Facteur du seuil robuste du filtre de pertes. |
| `anomaly_filter_history_size` | `10000` | Nombre maximal de pertes historiques conservées. |
| `anomaly_filter_min_history` | `1000` | Historique minimal pour appliquer ce filtre. |
| `initial_lr` | `0.0001` | Taux initial ; également réappliqué explicitement au chargement pour resume/evaluate dans omni-run. |
| `lr_anneal_factor` | `0.5` | Facteur multiplicatif de StepLR. |
| `lr_anneal_epoch_freq` | `10` | Période en époques de StepLR ; une valeur <= 0 désactive le scheduler. |
| `l2_reg` | `0.0001` | Weight decay de l’optimiseur. |
| `gradient_clip_norm` | `50` | Seuil de la norme du gradient, ou null. |
| `early_stop` | `true` | Active le nombre d’époques sans amélioration choisi ci-dessous. |
| `early_stopping_validation_patience` | `10` | Époques consécutives sans amélioration de la perte de validation avant arrêt. |
| `valid_portion` | `0.3` | Fraction finale du train réservée à validation ; son découpage précède l’ajustement du normaliseur. |
| `test_n_z` | `16` | Tirages latents pour les scores historiques de omni-run. |
| `test_batch_size` | `32` | Lot d’inférence pour omni-run. |
| `max_test_size` | `null` | Nombre maximal de lignes test ; null conserve toute la suite. |
| `get_score_on_dim` | `false` | Conserve les scores par caractéristique ; omni-spot effectue ensuite leur somme. |
| `bf_search_min` | `-400.0` | Borne basse de recherche supervisée de seuil, uniquement si des étiquettes sont fournies. |
| `bf_search_max` | `400.0` | Borne haute de recherche supervisée de seuil. |
| `bf_search_step_size` | `1.0` | Pas servant à définir le nombre de seuils supervisés évalués. |
| `level` | `0.01` | Paramètre historique en queue basse pour pot_eval de omni-run ; distinct du --level de omni-spot. |
| `pot_q` | `0.001` | Paramètre POT historique de omni-run, uniquement dans son évaluation avec étiquettes. |
| `device` | `null` | cpu, cuda, etc. ; null choisit CUDA si disponible, sinon CPU. Pour omni-spot, utiliser --device pour imposer le périphérique. |
| `save_z` | `false` | Exporte les latents via save_z en fichiers texte train_z_* / test_z_* dans le répertoire courant. |
| `save_dir` | `"model"` | Dossier des checkpoints, relatif au YAML ; null désactive leur sauvegarde pendant le run. |
| `result_dir` | `"result"` | Dossier des résultats, relatif au YAML. |
| `train_score_filename` | `"train_score.pkl"` | Nom de la sortie train sous result_dir ; null désactive cette sortie. |
| `test_score_filename` | `"test_score.pkl"` | Nom de la sortie test sous result_dir ; null désactive cette sortie. |
| `metrics_filename` | `"metrics.json"` | Nom du JSON des métriques sous result_dir. |

## `visualize_config.yaml`

| Clé | Valeur fournie | Usage |
|---|---|---|
| `dataset` | `"PMU"` | Conserver PMU ; le chargeur actuel lit le CSV PMU et contrôle les 14 grandeurs physiques. |
| `data_dir` | `"../data/pmu.csv"` | CSV séparé par ;, chemin relatif au YAML. |
| `data_column_order` | `["DFDT", "FREQ", "I0.ANG", "I0.MAG", "I1.ANG", "I1.MAG", "I2.ANG", "I2.MAG", "V0.ANG", "V0.MAG", "V1.ANG", "V1.MAG", "V2.ANG", "V2.MAG"]` | Noms ou suffixes ordonnés des 14 grandeurs physiques ; pas les colonnes sinus/cosinus du modèle. |
| `x_dim` | `null` | Dimension dérivée à partir de la représentation ; ne pas la fixer manuellement. |
| `exclude_zero_sequence_phase` | `false` | Exclut les phases des séquences homopolaires I0/V0, en conservant leurs magnitudes. |
| `excluded_features` | `[]` | Liste de grandeurs configurées à retirer de la représentation ; le chargeur attend toujours ses colonnes physiques. |
| `normalization_mode` | `"train_minmax"` | train_minmax, nominal ou window_local (min–max par fenêtre). |
| `nominal_normalization` | `{"DFDT": {"center": 0.0, "scale": 1.0}, "FREQ": {"center": 50.0, "scale": 0.5}, "I0.MAG": {"center": 0.0, "scale": 1.0}, "I1.MAG": {"center": 0.0, "scale": 1.0}, "I2.MAG": {"center": 0.0, "scale": 1.0}, "V0.MAG": {"center": 0.0, "scale": 240000.0}, "V1.MAG": {"center": 0.0, "scale": 240000.0}, "V2.MAG": {"center": 0.0, "scale": 240000.0}}` | Constantes center/scale par suffixe de grandeur. Appliquées aux grandeurs directes et magnitudes ; sinus/cosinus ont centre 0, échelle 1. |
| `use_connected_z_q` | `true` | Active le posterior latent récurrent conditionné par le latent précédent. |
| `use_connected_z_p` | `true` | Active le prior latent connecté (marche aléatoire gaussienne). |
| `z_dim` | `3` | Dimension latente. |
| `rnn_num_hidden` | `500` | Taille cachée des réseaux GRU. |
| `window_length` | `100` | Nombre de points par fenêtre ; doit correspondre au checkpoint. |
| `dense_dim` | `500` | Largeur des couches denses. |
| `std_epsilon` | `0.005` | Plancher de sigma pour les distributions à écart-type appris. |
| `decoder_sigma_mode` | `"fixed"` | fixed ou learned. |
| `decoder_fixed_sigma` | `0.05` | Écart-type du décodeur en mode fixed. |
| `get_score_on_dim` | `false` | Conserve les scores par caractéristique ; omni-spot effectue ensuite leur somme. |
| `train_start` | `0` | Indice de la première ligne de la partie train. |
| `max_train_size` | `0.8` | Fraction si flottant <= 1, sinon nombre de lignes ; null prend tout le reste et laisse le test vide. |
| `max_test_size` | `null` | Nombre maximal de lignes test ; null conserve toute la suite. |
| `checkpoint` | `"model/best_omni_anomaly.pt"` | Checkpoint de visualisation, relatif au YAML si non surchargé par la CLI. |
| `split` | `"test"` | train ou test pour la reconstruction continue. |
| `start` | `99` | Indice du premier point final demandé dans cette partition. |
| `points` | `1000` | Nombre de positions finales demandées ; les positions contenant des exclusions peuvent être absentes. |
| `batch_size` | `32` | Fenêtres par lot d’entraînement ; les lots incomplets sont écartés. |
| `n_z` | `16` | Tirages latents pour la reconstruction continue. |
| `device` | `null` | cpu, cuda, etc. ; null choisit CUDA si disponible, sinon CPU. Pour omni-spot, utiliser --device pour imposer le périphérique. |
| `backend` | `"matplotlib"` | matplotlib (.png) ou plotly (.html). |
| `output` | `null` | Chemin de figure relatif au répertoire courant ; null produit un nom automatique. |

## Normalisation et reprise

`train_minmax` ajuste les extrema sur les lignes valides de la partie de fitting,
après séparation de la validation. `nominal` utilise les références configurées,
avec une échelle strictement positive. `window_local` utilise les minima/maxima
de chaque fenêtre et conserve le contexte nécessaire à l’inversion. Changer le
mode au chargement d’un checkpoint est refusé.

La fenêtre, l’architecture, les noms et l’ordre des features, les exclusions et
les constantes nominales doivent correspondre au checkpoint. Les contrôles de
compatibilité sont conservés ; une erreur n’entraîne pas un réentraînement caché.

`omni-spot` lit ce YAML pour les données et le modèle, mais ses paramètres de
scoring (batch-size, n-z, stride, q, level, seed, device) viennent de sa CLI. Les
paramètres d’entraînement du YAML ne déclenchent aucun apprentissage dans cette
commande.
