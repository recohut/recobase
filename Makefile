setup:
	pip install -e .

test:
	PYTHONPATH=. pytest

ingest_ml1m:
	mkdir -p ./data/bronze/ml-1m
	dvc import -o ./data/bronze/ml-1m/movies.dat \
	https://github.com/sparsh-ai/reco-data \
	ml1m/v0/movies.dat
	dvc import -o ./data/bronze/ml-1m/users.dat \
	https://github.com/sparsh-ai/reco-data \
	ml1m/v0/users.dat
	dvc import -o ./data/bronze/ml-1m/ratings.dat \
	https://github.com/sparsh-ai/reco-data \
	ml1m/v0/ratings.dat