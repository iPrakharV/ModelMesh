.PHONY: lint compile test check train bench replay compose compose-prod smoke docker-build

lint:
	python -m ruff check .

compile:
	python -m compileall -q gateway worker training bench scripts tests

test:
	python -m pytest -q

check: lint compile test

train:
	python training/train_text_model.py

bench:
	python bench/run_scenarios.py

replay:
	python bench/replay_traffic.py --seconds-per-stage 8 --rps 18

compose:
	docker compose -f deploy/compose/docker-compose.yml up --build

compose-prod:
	docker compose -f deploy/compose/docker-compose.prod.yml up --build

smoke:
	python scripts/smoke_check.py

docker-build:
	docker build -f gateway/Dockerfile -t modelmesh-gateway:local .
	docker build -f worker/Dockerfile -t modelmesh-worker:local .
