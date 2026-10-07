# Validation du regroupement

Vérification effectuée le **7 octobre 2026**, sur Linux, avec Python 3.12.14 et
PyTorch CPU. Les versions source exactes figurent dans `SOURCES.json`.

## Contrôles exécutés

| Contrôle | Résultat |
|---|---|
| Installation editable des trois sous-projets dans un même environnement | Réussie |
| `python -m pip check` | Aucune dépendance déclarée manquante ou incompatible |
| `python -m pytest` depuis la racine | **98 tests réussis** |
| Vérification des noms non définis et erreurs de syntaxe (Ruff F821/F822/F823/E9) | Réussie |
| Aide de tous les points d'entrée et sous-commandes documentés | Réussie ; référence générée dans chaque `docs/CLI_REFERENCE.md` |
| Appels directs et `python -m` des paquets/scripts | Réussis |
| Chaîne IF des commandes de démonstration du README | CSV → fenêtres → caractéristiques → partitions → entraînement → scores → POT → rapport Plotly → classement/corrélations : réussie |
| Explications SHAP sur deux fenêtres de cette démonstration | SVG et tables produits ; résidu maximal d'additivité ≈ `1.11e-16` |
| Démonstration RPAD | 27 fenêtres évaluées, 2 fenêtres détectées, JSONL et rapport HTML produits |
| Petit entraînement OmniAnomaly, une époque sur données synthétiques | Checkpoints sauvegardés, meilleur état restauré et métriques produites |
| `omni-run --run-mode evaluate` sur ce checkpoint | Réussi, sans réentraînement |
| `omni-spot` sur ce checkpoint | 120 fenêtres test scorées, classement CSV et galerie SVG/NPZ produits |
| `omni-reconstruct --backend plotly` sur ce checkpoint | Rapport HTML produit |

Les nombres d'alarmes de ces exemples décrivent uniquement leurs données
synthétiques. Ils ne sont pas des mesures de précision/rappel sur un réseau.

## Comparaison avec la source Isolation Forest

Une exécution séparée a utilisé le code original du commit
`b95d4d9cf67fb9629fb0f054beadbcd97d52c6a4`, sans modification, pour préparer des
fenêtres et caractéristiques, entraîner une petite forêt et sauvegarder son
pickle. La version extraite a ensuite :

1. recalculé les fenêtres et caractéristiques à partir des mêmes mesures ;
2. chargé le **pickle créé par le code original** ;
3. recalculé les scores.

Les tables de fenêtres et de caractéristiques sont identiques, et les scores
sont **strictement identiques élément par élément**, sans tolérance numérique.
Cela valide la conservation du traitement sur ce cas et des chemins Python des
classes sauvegardées. Cela ne garantit pas le chargement de tout pickle créé
avec une autre version de scikit-learn ou un autre backend.

## Corrections constatées pendant la validation

- Argument `normalizer` inutilisé retiré des diagnostics `SigmaEpochStatistics` ;
  le test de la source signalait déjà ce problème.
- Référence à un `config` inexistant corrigée lors de la restauration du meilleur
  checkpoint en fin de `Trainer.fit` ; un test de sauvegarde/restauration après
  apprentissage protège maintenant ce parcours.
- Dates CSV OmniAnomaly à précision fractionnaire variable acceptées par un
  parsing explicite et conversion UTC.
- Chemins YAML/CLI OmniAnomaly harmonisés et couverts par un test ; les conventions
  exactes sont décrites dans `MIGRATION.md`.

## Environnement utilisé

| Dépendance | Version |
|---|---|
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| pyarrow | 25.0.1 |
| Plotly | 7.1.0 |
| SHAP | 0.52.0 |
| Matplotlib | 3.10.8 |
| PyTorch | 2.14.1+cpu |
| pyro-ppl | 1.9.2 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |

Ces versions décrivent l'environnement de vérification ; elles ne doivent pas
remplacer aveuglément celles ayant servi à créer les modèles existants.

## Limites de cette validation

- Aucun modèle de production ni mesure privée n'était fourni dans ces branches ;
  leur compatibilité et leurs performances n'ont pas été testées.
- Aucun essai CUDA/cuML, Docker ou sous Windows n'a été exécuté. Les commandes
  PowerShell sont documentées, et les chemins sont traités avec `pathlib`.
- Aucune mesure de temps réel sur une machine de déploiement n'a été réalisée.
- La forêt conserve son seuil POT fixe ; ce nettoyage ne le transforme pas en
  SPOT dynamique. Les conventions distinctes d'OmniAnomaly restent documentées.

Les données, poids et rapports générés pendant les essais ne sont pas inclus dans
le dépôt regroupé. Les tests et générateurs livrés permettent de reproduire les
vérifications courantes. Pour relancer la suite :

```bash
python -m pip install -e "./isolation_forest[shap,test]" -e "./rpad_detection[test]" -e "./omnis[test]"
python -m pytest
```

## Ajout du classement IF autonome — 7 octobre 2026

Le script `isolation_forest/rank_windows.py` a été exécuté sur les **400 fenêtres**
de scores d'évaluation de la démonstration IF. Son CSV conserve les mêmes scores,
identifiants et dates, dans le même ordre décroissant que le rapport d'analyse
existant, avec un rang supplémentaire commençant à 1.

L'option `--top-n 20` produit les 20 premières lignes du classement complet.
Une entrée vide conserve les en-têtes ; un artefact de décisions SPOT est refusé
avec une erreur explicite. Les empreintes du Parquet des scores et de son manifeste
sont identiques avant et après l'exécution. L'aide `--help` a été vérifiée et ajoutée
à la référence des commandes. Ces contrôles portent sur le nouveau script ; les
98 tests ci-dessus décrivent la validation initiale du regroupement.
