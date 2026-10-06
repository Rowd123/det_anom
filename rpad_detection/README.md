# Détection d'événements par différence de phase entre deux PMU

Toutes les commandes partent de **`det_anom/`**.
Installation : `python -m pip install -e ./rpad_detection`.

Ce module analyse deux angles PMU synchronisés. Il déroule chaque angle, calcule
leur différence (RPAD), puis applique des critères FFT, Matrix Pencil,
Yule-Walker et min–max. Les critères dépassant leur seuil votent pour une alarme.
Les deux amplitudes Matrix Pencil comptent pour **une seule méthode**.

Il ne génère pas de signaux triphasés, n'estime pas de phaseurs et n'effectue pas
de classification des événements. La détection, son état de flux et les rapports
sont indépendants de l'ancien simulateur.

## Données d'entrée

Chaque ligne doit fournir les angles de **deux PMU au même instant**. Les colonnes
`angle_a` et `angle_b` ne désignent pas les phases A et B d'une même PMU.
Les deux mesures doivent déjà être synchronisées ; il n'y a pas de jointure
automatique de deux fichiers de stations.

CSV en secondes :

```csv
time,angle_a,angle_b,valid
0.00,179.2,174.1,1
0.02,179.3,174.2,1
0.04,,,0
```

Dans cet exemple à 50 Hz, régler `sampling_rate: 50` dans la configuration.
Le fichier fourni par défaut est réglé à **30 Hz** : l'adapter à la cadence réelle.
Les flags facultatifs sont `1`/`0` ou `true`/`false`. Une case angulaire vide est
manquante. Un angle égal à zéro est une mesure valide.

Les timestamps doivent être croissants, sur la grille configurée. Le premier
doublon est conservé ; une entrée hors ordre ou hors grille est refusée.
Les angles sont exprimés dans l'unité de `angle_unit` (`degrees` ou `radians`).

Les dates ISO 8601 sont également acceptées avec `--time-format iso8601` :

```csv
DATE_MESURE;angle_a;angle_b;valid
2026-09-01T00:00:00Z;179.2;174.1;1
2026-09-01T00:00:00.020Z;179.3;174.2;1
```

Le fuseau est obligatoire (`Z` ou décalage comme `+02:00`). Ces dates deviennent
des **secondes Unix** dans les décisions ; les intervalles d'exclusion doivent
alors utiliser cette même origine. L'API de flux reçoit toujours des secondes
numériques, pas des chaînes de dates.

## Détection hors ligne sur un CSV

```bash
rpad-detection detect-phase --config rpad_detection/examples/rpad_detection.json --csv data/angles.csv --time-column time --angle-a angle_a --angle-b angle_b --valid-column valid --sep "," --output artifacts/rpad/decisions.jsonl --report-dir artifacts/rpad/report
```

Exemple pour un CSV daté séparé par des points-virgules :

```bash
rpad-detection detect-phase --config rpad_detection/examples/rpad_detection.json --csv data/angles_dated.csv --time-column DATE_MESURE --time-format iso8601 --angle-a angle_a --angle-b angle_b --valid-column valid --sep ";" --output artifacts/rpad/decisions.jsonl --report-dir artifacts/rpad/report
```

`--valid-column` s'omet si le fichier n'a pas de flag. `--report-dir` est
facultatif. La détection sauvegarde les décisions et un fichier de configuration
`decisions.jsonl.config.json`. Le calcul hors ligne utilise le même détecteur
que le flux et appelle `flush()` à la fin pour traiter l'état restant.

Pour régénérer uniquement les rapports :

```bash
rpad-detection phase-report artifacts/rpad/decisions.jsonl artifacts/rpad/report
```

Ouvrir `artifacts/rpad/report/index.html`. L'index contient les courbes des critères
et de leurs seuils. Chaque fenêtre détectée dispose d'une page `event-XXXXX.html`
avec son déphasage, les valeurs des critères, les références, seuils, dépassements,
votes et paramètres estimés. Conserver tout le dossier, y compris `plotly.min.js` ;
les pages fonctionnent hors ligne.

## Configuration des décisions

`rpad_detection/examples/rpad_detection.json` est la configuration complète pour
les modes causaux. `rpad_retrospective.json` illustre l'analyse par bloc hors ligne.
Toutes les clés sont contrôlées ; une clé inconnue est refusée.

| Paramètre | Rôle |
|---|---|
| `sampling_rate`, `angle_unit`, `pair_id` | Cadence des rapports, unité angulaire, identifiant de la paire |
| `window_samples`, `hop_samples` | Longueur et pas des fenêtres, en échantillons |
| `preprocessing.mu`, `sigma` | Constantes de `(RPAD - mu) / sigma`, après conversion en degrés ; `sigma > 0` |
| `preprocessing.max_interpolation_gap` | Longueur maximale d'une lacune interne interpolable |
| `preprocessing.min_valid_fraction` | Fraction minimale réellement observée dans une fenêtre |
| `preprocessing.timestamp_tolerance` | Tolérance d'alignement, en secondes |
| `preprocessing.excluded_intervals` | Intervalles `[début, fin]` inclusifs en secondes, jamais interpolés |
| `methods` | Sous-ensemble de `fft`, `matrix_pencil`, `yule_walker`, `minmax` |
| `frequency_band` | Bande analysée, en Hz ; bornes utilisées : bas exclu, haut inclus |
| `fft.preprocess`, `window`, `nfft` | Prétraitement propre au critère, fenêtre et taille FFT |
| `matrix_pencil.preprocess`, `pencil_length` | Prétraitement et dimension du crayon |
| `matrix_pencil.rank` | Rang imposé ; `null` active la sélection par tolérance SVD |
| `matrix_pencil.svd_relative_tolerance`, `max_rank`, `rcond` | Tolérance, rang plafond et seuil de résolution des moindres carrés |
| `yule_walker.preprocess`, `order`, `nfft` | Prétraitement, ordre AR et grille spectrale |
| `yule_walker.ridge`, `power_floor` | Régularisation diagonale et plancher avant le logarithme |
| `threshold.mode` | `rolling`, `frozen` ou `retrospective` |
| `threshold.history_windows`, `min_history` | Taille maximale de référence et minimum pour évaluer |
| `threshold.ddof`, `k_sigma`, `min_std` | Correction d'écart-type, facteur de seuil et plancher numérique |
| `threshold.update_policy` | `all` ou `normal_only` pour la référence glissante |
| `threshold.reset_on_gap` | Réinitialisation de la référence après une interruption |
| `consensus.min_methods` | Nombre de méthodes distinctes nécessaires pour déclencher |

Les prétraitements de critères disponibles sont `none`, `constant`, `linear` et
`difference`. Ce sont des opérations propres aux critères, distinctes de la
normalisation fixe du RPAD. Un changement de longueur/cadence doit aussi respecter
les contraintes de rang, d'ordre AR et de taille FFT contrôlées par le code.

| Mode | Référence utilisée | Flux possible ? |
|---|---|---|
| `rolling` | Fenêtres strictement antérieures, mémoire bornée | Oui |
| `frozen` | Les `min_history` premières fenêtres valides | Oui |
| `retrospective` | Toutes les fenêtres valides du fichier fourni | Non |

Le critère est comparé à la moyenne de référence plus `k_sigma` fois l'écart-type,
avec son plancher numérique. Une fenêtre `warmup` ou rejetée n'est **pas** une
fenêtre déclarée normale. Les valeurs standard `mu=0`, `sigma=1` sont une identité
explicitement choisie pour le déphasage en degrés, pas un ajustement statistique.

Conventions numériques conservées : FFT unilatérale corrigée du gain cohérent ;
différences finies non divisées par la période ; amplitudes Matrix Pencil égales
aux modules des résidus complexes (A/2 pour un cosinus réel d'amplitude A) ;
Yule-Walker fondé sur une autocorrélation biaisée, PSD unilatérale et `10 log10`.
Le RPAD non normalisé est conservé en degrés dans les explications.

## Traitement en flux et reprise

Chaque ligne NDJSON est une observation autonome :

```json
{"timestamp": 10.0, "angle_a": 179.2, "angle_b": 174.1, "valid": true}
```

Sous Bash, créer le dossier avant la redirection puis lancer :

```bash
mkdir -p artifacts/rpad
rpad-detection phase-stream --config rpad_detection/examples/rpad_detection.json --state-out artifacts/rpad/live_state.json --checkpoint-every 1000 < data/angles.jsonl > artifacts/rpad/live_decisions.jsonl
```

Sous PowerShell :

```powershell
New-Item -ItemType Directory -Force artifacts/rpad
Get-Content data/angles.jsonl | rpad-detection phase-stream --config rpad_detection/examples/rpad_detection.json --state-out artifacts/rpad/live_state.json --checkpoint-every 1000 | Set-Content -Encoding utf8 artifacts/rpad/live_decisions.jsonl
```

En exploitation, remplacer la lecture de fichier par un producteur d'observations
synchronisées. Les décisions sont émises sur stdout dès qu'elles sont disponibles.

Reprise sur le **suffixe non traité**, sous Bash :

```bash
rpad-detection phase-stream --config rpad_detection/examples/rpad_detection.json --state-in artifacts/rpad/live_state.json --state-out artifacts/rpad/live_state.json < data/next_chunk.jsonl >> artifacts/rpad/live_decisions.jsonl
```

Sous PowerShell, remplacer `Set-Content` par `Add-Content -Encoding utf8` et
alimenter la commande avec `Get-Content data/next_chunk.jsonl`.

L'état JSON contient fenêtres partielles, lacunes en attente, déroulement angulaire,
référence, compteurs et configuration. Une reprise avec configuration différente
est refusée. Utiliser un état par paire. La fermeture du flux conserve les
fenêtres partielles ; elle ne les force pas à devenir des fenêtres complètes.
La sortie et le checkpoint ne forment pas une transaction : le système
d'acquisition doit conserver ses offsets et gérer les doublons après un arrêt.

## API Python

```python
from pmu_simulator.detection.config import load_config
from pmu_simulator.detection.stream import PhaseEventDetector

config = load_config("rpad_detection/examples/rpad_detection.json")
detector = PhaseEventDetector(config)
decisions = detector.update(timestamp=0.0, angle_a=179.2, angle_b=174.1, valid=True)
detector.save_state("artifacts/rpad/live_state.json")
restored = PhaseEventDetector.load_state("artifacts/rpad/live_state.json", config)
```

`update` renvoie une liste éventuellement vide. Pour une relecture hors ligne,
`replay(samples, config)` et `save_decisions(rows, path, config)` se trouvent dans
`pmu_simulator.detection.io`.

## Conteneur et démonstration

Construire l'image avec le sous-répertoire comme contexte :

```bash
docker build -t pmu-rpad-detector ./rpad_detection
```

Exemple Bash, avec montage de la configuration et de l'état :

```bash
mkdir -p artifacts/rpad
docker run --rm -i -v "$PWD/rpad_detection/examples/rpad_detection.json:/config/detector.json:ro" -v "$PWD/artifacts/rpad:/state" pmu-rpad-detector --config /config/detector.json --state-out /state/live_state.json < data/angles.jsonl > artifacts/rpad/live_decisions.jsonl
```

Le conteneur expose `phase-stream`. Docker n'est pas requis pour une exécution
Python locale. Les paramètres Matrix Pencil peuvent représenter la majorité du
temps de calcul ; mesurer `compute_seconds` et `compute_over_hop` sur la machine
cible. Le délai comprend aussi la fenêtre et l'attente de l'échantillon qui borne
une petite lacune. Il ne s'agit pas d'une garantie de temps réel dur.

```bash
python rpad_detection/scripts/demo_phase_detection.py --output artifacts/rpad_demo
python -m pytest rpad_detection/tests
```

La démonstration génère deux séries d'angles synthétiques synchronisées, bruitées,
avec une oscillation amortie et un saut. Elle produit `angles.csv`, `angles.jsonl`,
`config.json`, `decisions.jsonl`, la copie de configuration et `report/index.html`.
Elle teste le détecteur sans réintroduire le simulateur de phaseurs.

## Aide

`rpad-detection --help` liste les trois commandes ; ajouter `--help` après une
sous-commande pour ses options. [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) rassemble
cette aide. `python -m pmu_simulator ...` est équivalent. L'alias `pmu-sim` est
conservé uniquement pour `detect-phase`, `phase-stream` et `phase-report`.
