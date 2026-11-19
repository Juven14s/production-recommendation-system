# Production Recommendation System

Système de recommandation de films en deux étapes : génération rapide de candidats,
puis ranking personnalisé. Le projet part de MovieLens et évoluera vers un service
FastAPI observable, versionné et réentraînable.

## État actuel

Le premier incrément couvre :

- l'ingestion de MovieLens au format CSV (20M/25M/32M) ou DAT (1M) ;
- la normalisation des colonnes et des types ;
- la validation du contrat de données ;
- l'écriture de tables Parquet ;
- la production d'un rapport JSON de qualité ;
- des tests unitaires et une CI minimale.

Le deuxième incrément ajoute :

- un split temporel leave-two-out par utilisateur ;
- une définition explicite des interactions positives ;
- une baseline de popularité qui exclut les films déjà vus ;
- Recall@K, NDCG@K et couverture du catalogue sur la validation.

Le troisième incrément ajoute un filtrage collaboratif Item-KNN implicite avec
similarité cosinus, shrinkage, voisinage sparse et fallback de popularité.

Le quatrième incrément ajoute une factorisation BPR en PyTorch et produit des
embeddings NumPy portables, prêts à être indexés dans FAISS.

Le cinquième incrément construit un index FAISS exact, conserve le biais du modèle
BPR dans le produit scalaire, mesure le rappel des candidats et la latence p95.

Le sixième incrément ajoute un ranker LightGBM LambdaRank avec génération de
données d'apprentissage sans fuite temporelle et features online-compatible.

Le septième incrément sert le pipeline complet avec FastAPI, version de modèle,
cache Redis avec fallback mémoire, cold start et conteneurs Docker Compose.

Le huitième incrément ajoute l'observabilité Prometheus, les alertes de latence et
d'erreur, ainsi qu'un dashboard Grafana provisionné automatiquement.

Le neuvième incrément ajoute MLflow, les bundles autonomes, les quality gates,
l'alias `champion` et un rollback atomique et auditable.

Le dixième incrément ajoute la CI/CD GitHub Actions : qualité et packaging,
construction GHCR, réentraînement planifié, validation d'intégrité du bundle,
promotion protégée vers staging/production et rollback d'une image immuable.

## Démarrage

Prérequis : Python 3.11 ou 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Placez les fichiers MovieLens dans `data/raw/ml-1m` ou `data/raw/ml-25m`, puis lancez :

```powershell
mle-ingest --input-dir data/raw/ml-1m --output-dir data/processed/ml-1m
```

La commande détecte automatiquement `ratings.csv`/`movies.csv` ou
`ratings.dat`/`movies.dat`. Elle écrit :

```text
data/processed/ml-1m/
├── ratings.parquet
├── movies.parquet
└── quality_report.json
```

Pour exécuter sans installer le script CLI :

```powershell
$env:PYTHONPATH = "src"
python -m mle_recsys.data.ingest `
  --input-dir data/raw/ml-1m `
  --output-dir data/processed/ml-1m
```

Créez ensuite le split temporel :

```powershell
mle-split `
  --ratings data/processed/ml-1m/ratings.parquet `
  --output-dir data/processed/ml-1m/splits `
  --positive-threshold 4.0
```

Puis entraînez et évaluez la baseline de popularité :

```powershell
mle-popularity `
  --split-dir data/processed/ml-1m/splits `
  --movies data/processed/ml-1m/movies.parquet `
  --output-dir artifacts/baselines/popularity `
  --k 5 10 20
```

Entraînez ensuite la baseline collaborative :

```powershell
mle-item-knn `
  --split-dir data/processed/ml-1m/splits `
  --movies data/processed/ml-1m/movies.parquet `
  --output-dir artifacts/baselines/item-knn `
  --neighbors-per-item 100 `
  --shrinkage 10 `
  --k 5 10 20
```

Apprenez les embeddings BPR :

```powershell
mle-bpr `
  --split-dir data/processed/ml-1m/splits `
  --movies data/processed/ml-1m/movies.parquet `
  --output-dir artifacts/models/bpr `
  --embedding-dim 64 `
  --epochs 15 `
  --batch-size 4096 `
  --k 5 10 20
```

Construisez et évaluez l'index FAISS :

```powershell
mle-faiss `
  --bpr-dir artifacts/models/bpr `
  --split-dir data/processed/ml-1m/splits `
  --movies data/processed/ml-1m/movies.parquet `
  --output-dir artifacts/retrieval/faiss `
  --k 50 100 200
```

Entraînez le ranker sur les candidats FAISS :

```powershell
mle-ranker `
  --split-dir data/processed/ml-1m/splits `
  --movies data/processed/ml-1m/movies.parquet `
  --bpr-dir artifacts/models/bpr `
  --validation-candidates artifacts/retrieval/faiss/validation_candidates.parquet `
  --output-dir artifacts/models/ranker `
  --candidate-count 200 `
  --k 5 10 20
```

Lancez l'API localement :

```powershell
mle-serve
```

Puis testez les endpoints :

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod "http://localhost:8000/recommendations/1?k=10"
```

Avec Docker et Redis :

```powershell
docker compose up --build
```

Services exposés par Compose :

- API et documentation : `http://localhost:8000/docs`
- métriques Prometheus brutes : `http://localhost:8000/metrics`
- serveur Prometheus : `http://localhost:9090`
- dashboard Grafana : `http://localhost:3000` (`admin` / `admin` en local)
- interface MLflow : `http://localhost:5000`

Réentraîner et promouvoir automatiquement si les quality gates passent :

```powershell
mle-retrain
```

Consulter les runs et le registre MLflow sans Docker :

```powershell
mlflow server `
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db `
  --default-artifact-root artifacts/mlflow/artifacts `
  --host 127.0.0.1 `
  --port 5000
```

Dans Compose, le profil d'entraînement journalise directement vers le serveur
MLflow et partage le registre avec l'API :

```powershell
docker compose --profile training run --rm retrainer
docker compose restart api
```

Inspecter le registre ou revenir au champion précédent :

```powershell
mle-registry status
mle-registry rollback
```

Valider explicitement un bundle avant déploiement :

```powershell
mle-validate-bundle artifacts/model_registry/bundles/<version>
```

## CI/CD GitHub

- `CI` exécute Ruff, les tests, le build Python et valide les configurations
  Prometheus/Grafana sur chaque push et pull request.
- `Container` construit l'image de base et la publie dans GHCR sur `main` ou un tag.
- `Retrain` s'exécute manuellement ou chaque lundi, vérifie le checksum officiel de
  MovieLens, restaure les métriques du champion précédent et publie un bundle versionné.
- `Promote model image` télécharge ce bundle, contrôle ses empreintes et son inférence,
  puis attend l'approbation du GitHub Environment `staging` ou `production`.
- `Roll back model image` remet un tag d'environnement sur une image immuable déjà
  testée, sans réentraîner le modèle.

L'image de release n'embarque pas PyTorch, CUDA ni MLflow : les artefacts BPR sont
chargés sous forme NumPy et seuls les composants nécessaires à l'inférence sont installés.

Dans les paramètres GitHub, créez les Environments `staging` et `production`, puis
ajoutez au minimum un reviewer obligatoire sur `production`. Les workflows utilisent
uniquement `GITHUB_TOKEN` et ne nécessitent donc aucun secret de registre additionnel.

## Principes retenus

- Les identifiants sont normalisés en `user_id` et `movie_id`.
- Les timestamps restent des entiers Unix pour préserver la source sans ambiguïté.
- L'ingestion échoue explicitement si le schéma, les types ou les domaines sont invalides.
- Les données et artefacts volumineux ne sont pas versionnés dans Git.
- Les découpages d'entraînement seront temporels afin d'éviter les fuites de données.

## Prochain incrément

Exécuter les workflows sur le dépôt GitHub, mesurer le comportement en charge de
l'API et ajouter un test de dérive sur des interactions plus récentes.
