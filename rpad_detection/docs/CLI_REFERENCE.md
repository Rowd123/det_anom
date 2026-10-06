# Référence complète des commandes

Aide générée depuis les parseurs de cette version. Les commandes partent de la racine `det_anom`.

## `rpad-detection`

```text
usage: rpad-detection [-h] {detect-phase,phase-stream,phase-report} ...

Détection d'événements sur une différence de phase entre deux PMU

positional arguments:
  {detect-phase,phase-stream,phase-report}
    detect-phase        Détection RPAD depuis deux angles synchronisés en CSV
    phase-stream        NDJSON stdin → décisions NDJSON stdout (causal)
    phase-report        Visualiser les décisions sauvegardées sans recalcul

options:
  -h, --help            show this help message and exit
```

## `rpad-detection detect-phase`

```text
usage: rpad-detection detect-phase [-h] --config CONFIG --csv CSV
                                   [--time-column TIME_COLUMN]
                                   [--angle-a ANGLE_A] [--angle-b ANGLE_B]
                                   [--valid-column VALID_COLUMN] [--sep SEP]
                                   [--time-format {seconds,iso8601}] --output
                                   OUTPUT [--report-dir REPORT_DIR]

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --csv CSV
  --time-column TIME_COLUMN
  --angle-a ANGLE_A
  --angle-b ANGLE_B
  --valid-column VALID_COLUMN
  --sep SEP
  --time-format {seconds,iso8601}
                        Format de la colonne temporelle; ISO 8601 exige un
                        fuseau explicite
  --output OUTPUT
  --report-dir REPORT_DIR
```

## `rpad-detection phase-stream`

```text
usage: rpad-detection phase-stream [-h] --config CONFIG [--state-in STATE_IN]
                                   [--state-out STATE_OUT]
                                   [--checkpoint-every CHECKPOINT_EVERY]

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --state-in STATE_IN
  --state-out STATE_OUT
  --checkpoint-every CHECKPOINT_EVERY
```

## `rpad-detection phase-report`

```text
usage: rpad-detection phase-report [-h] decisions output_dir

positional arguments:
  decisions
  output_dir

options:
  -h, --help  show this help message and exit
```

