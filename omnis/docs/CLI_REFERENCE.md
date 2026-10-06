# Référence complète des commandes

Aide générée depuis les parseurs de cette version. Les commandes partent de la racine `det_anom`.

## `omni-run`

```text
usage: omni-run [-h] [--config CONFIG] [--run-mode {scratch,resume,evaluate}]
                [--checkpoint CHECKPOINT]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Chemin vers le fichier YAML
  --run-mode {scratch,resume,evaluate}
                        Surcharge run_mode du YAML
  --checkpoint CHECKPOINT
                        Checkpoint relatif au répertoire courant; pour
                        resume/evaluate
```

## `omni-spot`

```text
usage: omni-spot [-h] [--config CONFIG] --checkpoint CHECKPOINT
                 [--output OUTPUT] [--top TOP] [--stride STRIDE]
                 [--batch-size BATCH_SIZE] [--n-z N_Z] [--q Q] [--level LEVEL]
                 [--seed SEED] [--device DEVICE]

Inference-only SPOT analysis and ranked physical window reconstructions.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration exacte du modèle entraîné
  --checkpoint CHECKPOINT
                        Chemin relatif au répertoire courant
  --output OUTPUT
  --top TOP             Fenêtres à reconstruire/exporter; 0 = toutes
  --stride STRIDE       Pas en échantillons; défaut = longueur de fenêtre
                        (sans recouvrement); 1 = ancien comportement
  --batch-size BATCH_SIZE
  --n-z N_Z
  --q Q
  --level LEVEL         Quantile initial en queue haute
  --seed SEED
  --device DEVICE
```

## `omni-reconstruct`

```text
usage: omni-reconstruct [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                        [--split {train,test}] [--start START]
                        [--points POINTS] [--batch-size BATCH_SIZE]
                        [--n-z N_Z] [--device DEVICE]
                        [--backend {matplotlib,plotly}] [--output OUTPUT]

Visualise les valeurs PMU physiques et leur reconstruction OmniAnomaly.
Exemples (depuis la racine du dépôt) : python
omnis/visualize_reconstruction.py python omnis/visualize_reconstruction.py
--backend plotly --split train

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --checkpoint CHECKPOINT
  --split {train,test}
  --start START
  --points POINTS
  --batch-size BATCH_SIZE
  --n-z N_Z
  --device DEVICE
  --backend {matplotlib,plotly}
  --output OUTPUT
```

