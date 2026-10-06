# Référence complète des commandes

Aide générée depuis les parseurs de cette version. Les commandes partent de la racine `det_anom`.

## `spectral-anomaly`

```text
usage: spectral-anomaly [-h] [--log-level LOG_LEVEL]
                        {train-iforest,score-iforest,spot,plot-spot,prepare-windows,extract-windows,split-windows,analyze-scores}
                        ...

positional arguments:
  {train-iforest,score-iforest,spot,plot-spot,prepare-windows,extract-windows,split-windows,analyze-scores}
    plot-spot           Plot original CSV and saved SPOT window decisions
    analyze-scores      Rank saved scores and inspect raw feature correlations

options:
  -h, --help            show this help message and exit
  --log-level LOG_LEVEL
```

## `spectral-anomaly prepare-windows`

```text
usage: spectral-anomaly prepare-windows [-h] --config CONFIG --source-id
                                        SOURCE_ID --channel-id CHANNEL_ID
                                        [--value-col VALUE_COL]
                                        [--index-col INDEX_COL] [--sep SEP]
                                        [--datetime-format DATETIME_FORMAT]
                                        input output

positional arguments:
  input
  output

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --source-id SOURCE_ID
  --channel-id CHANNEL_ID
  --value-col VALUE_COL
  --index-col INDEX_COL
  --sep SEP
  --datetime-format DATETIME_FORMAT
```

## `spectral-anomaly extract-windows`

```text
usage: spectral-anomaly extract-windows [-h] --config CONFIG input output

positional arguments:
  input
  output

options:
  -h, --help       show this help message and exit
  --config CONFIG
```

## `spectral-anomaly split-windows`

```text
usage: spectral-anomaly split-windows [-h] --config CONFIG input output_dir

positional arguments:
  input
  output_dir

options:
  -h, --help       show this help message and exit
  --config CONFIG
```

## `spectral-anomaly train-iforest`

```text
usage: spectral-anomaly train-iforest [-h] --config CONFIG features model

positional arguments:
  features
  model

options:
  -h, --help       show this help message and exit
  --config CONFIG
```

## `spectral-anomaly score-iforest`

```text
usage: spectral-anomaly score-iforest [-h] [--batch-size BATCH_SIZE]
                                      [--resume]
                                      features model output

positional arguments:
  features
  model
  output

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
  --resume
```

## `spectral-anomaly spot`

```text
usage: spectral-anomaly spot [-h] --config CONFIG [--resume]
                             calibration scores output state

positional arguments:
  calibration
  scores
  output
  state

options:
  -h, --help       show this help message and exit
  --config CONFIG
  --resume
```

## `spectral-anomaly plot-spot`

```text
usage: spectral-anomaly plot-spot [-h] --source-id SOURCE_ID --channel-id
                                  CHANNEL_ID [--value-col VALUE_COL]
                                  [--index-col INDEX_COL] [--sep SEP]
                                  [--datetime-format DATETIME_FORMAT]
                                  input decisions output

positional arguments:
  input
  decisions
  output

options:
  -h, --help            show this help message and exit
  --source-id SOURCE_ID
  --channel-id CHANNEL_ID
  --value-col VALUE_COL
  --index-col INDEX_COL
  --sep SEP
  --datetime-format DATETIME_FORMAT
```

## `spectral-anomaly analyze-scores`

```text
usage: spectral-anomaly analyze-scores [-h] [--model MODEL] [--top-n TOP_N]
                                       features scores output

positional arguments:
  features
  scores
  output

options:
  -h, --help     show this help message and exit
  --model MODEL
  --top-n TOP_N
```

## `spectral-anomaly-shap`

```text
usage: spectral-anomaly-shap [-h] --train TRAIN [--top-n TOP_N]
                             [--window-id WINDOW_ID] [--date DATE]
                             [--source-id SOURCE_ID] [--channel-id CHANNEL_ID]
                             [--background-size BACKGROUND_SIZE]
                             [--permutations PERMUTATIONS] [--seed SEED]
                             [--batch-size BATCH_SIZE]
                             model observations output

Post-hoc SHAP of the saved pipeline's exact higher-is-atypical score.

positional arguments:
  model
  observations          evaluation or new window features Parquet (not scores)
  output

options:
  -h, --help            show this help message and exit
  --train TRAIN         training feature artifact with its manifest
  --top-n TOP_N
  --window-id WINDOW_ID
  --date DATE           exact window_start; timezone-naive values interpreted
                        as UTC
  --source-id SOURCE_ID
  --channel-id CHANNEL_ID
  --background-size BACKGROUND_SIZE
  --permutations PERMUTATIONS
  --seed SEED
  --batch-size BATCH_SIZE
```

