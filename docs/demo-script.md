# Script de démonstration : 5 minutes

## 0:00–0:30 : Problème et résultat

> J'ai construit un système de recommandation de films en deux étapes, depuis les
> interactions MovieLens brutes jusqu'à une image de production observable et
> réversible. Le système utilise FAISS pour récupérer 200 candidats puis LightGBM pour
> les classer. Le champion est réentraîné, validé et promu par GitHub Actions.

À montrer : le haut du `README.md`, les badges verts et le tableau des résultats.

## 0:30–1:30 : Architecture

À montrer : le diagramme Mermaid du README.

Points à expliquer :

1. Le split est temporel afin de simuler une recommandation future.
2. BPR apprend les embeddings utilisateur/film.
3. FAISS réduit le catalogue à 200 candidats en faible latence.
4. LightGBM LambdaRank combine les signaux d'embedding, popularité, activité,
   récence et affinité de genres.
5. Un fallback de popularité traite les utilisateurs inconnus.

## 1:30–2:15 : Résultats et compromis

> Le champion atteint 0,0767 de Recall@10, 0,0365 de NDCG@10, 46,07 % de couverture
> et 53,31 % de candidate recall@200. Ces valeurs sont modestes en absolu parce que
> l'évaluation impose un unique film futur pertinent parmi plusieurs milliers. Le point
> important est que les seuils sont explicites, versionnés et bloquants.

Compromis à mentionner :

- FAISS `IndexFlatIP` privilégie ici l'exactitude et la simplicité ; un index ANN
  deviendrait pertinent avec un catalogue beaucoup plus grand.
- MovieLens ne fournit pas de contexte online réel ; les métriques offline ne remplacent
  donc pas un A/B test.
- Le retrieval p95 mesuré ne représente pas la latence HTTP end-to-end.

## 2:15–3:15 : Démonstration API

```powershell
docker compose up --build
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod "http://localhost:8000/recommendations/1?k=5"
Invoke-RestMethod "http://localhost:8000/recommendations/999999?k=5"
```

À faire remarquer :

- la version du modèle est présente dans chaque réponse ;
- les films déjà vus sont exclus ;
- `cold_start` passe à `true` pour l'utilisateur inconnu ;
- le second appel identique peut être servi par Redis.

## 3:15–4:15 : MLOps et observabilité

Ouvrir les runs GitHub Actions de référence depuis le README.

> Un tag de réentraînement lance ingestion, validation, BPR, FAISS, LightGBM et MLflow.
> Le candidat ne devient champion que si les quatre quality gates passent. La promotion
> recharge ensuite le bundle, vérifie ses SHA-256, exécute une inférence connue et cold
> start, construit l'image, appelle `/health`, puis publie un tag immuable. Production
> peut exiger une approbation humaine et le rollback ne fait que restaurer une image
> déjà testée.

À montrer localement si le temps le permet :

- MLflow : paramètres, métriques et alias `champion` ;
- Grafana : latence, erreurs, cache, scores et fraîcheur des features ;
- `manifest.json` : version, fingerprints, métriques et résultat des gates.

## 4:15–5:00 : Conclusion

> Ce projet démontre moins un algorithme isolé qu'un cycle de vie ML complet : données
> validées, évaluation sans fuite, artefacts reproductibles, service observable,
> déploiement contrôlé et rollback. Les prolongements naturels seraient la dérive, les
> tests de charge, l'A/B testing et un feature store partagé.

## Questions probables en entretien

### Pourquoi ne pas recommander directement avec LightGBM sur tout le catalogue ?

Le coût des features et du scoring croît avec le nombre de films. Le retrieval réduit
l'espace de recherche ; le ranker concentre ensuite le calcul sur les candidats utiles.

### Comment évites-tu la fuite temporelle ?

Le split réserve les interactions futures. Le dataset du ranker possède en plus son
propre retriever interne, entraîné uniquement sur l'historique autorisé à cette étape.

### Que se passe-t-il si Redis tombe ?

L'API utilise un cache mémoire TTL comme fallback. La recommandation reste disponible,
mais le cache n'est plus partagé entre plusieurs réplicas.

### Comment détecter un mauvais modèle qui passe les tests ?

Les gates absolues sont complétées par une comparaison au champion précédent. En
production réelle, j'ajouterais des métriques métier online, une analyse de dérive et un
déploiement progressif avec A/B test ou canary.

### Pourquoi un bundle plutôt que des fichiers indépendants ?

Le bundle garantit que données de référence, embeddings, index, ranker et manifeste
restent cohérents. Les fingerprints empêchent de déployer un mélange de versions.

### Comment passer à plusieurs millions de films ?

Je remplacerais l'index exact par HNSW ou IVF/PQ, externaliserais les features dans un
feature store, servirais plusieurs réplicas et mesurerais le compromis rappel/latence.
