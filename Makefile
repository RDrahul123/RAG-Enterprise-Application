.PHONY: help install corpus ingest run test eval docker clean pull-model

help:
	@echo "make install     install dependencies into .venv"
	@echo "make corpus      generate the sample PDF corpus"
	@echo "make ingest      build the vector + BM25 index"
	@echo "make run         start the API and UI on :8000"
	@echo "make test        run the test suite (no model downloads)"
	@echo "make eval        run the retrieval/answer evaluation"
	@echo "make pull-model  pre-download the local LLM"
	@echo "make docker      build and run with docker compose"
	@echo "make clean       remove the index and caches"

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

corpus:
	.venv/bin/python scripts/make_sample_corpus.py

ingest:
	.venv/bin/python scripts/ingest.py

run:
	.venv/bin/uvicorn app.main:app --reload --port 8000

test:
	RAG_USE_REAL_MODELS=0 .venv/bin/python -m pytest tests -q

eval:
	.venv/bin/python evals/run_eval.py --k 5

pull-model:
	.venv/bin/python -c "from transformers import pipeline; \
		pipeline('text-generation', model='Qwen/Qwen2.5-1.5B-Instruct')"

docker:
	docker compose up --build

clean:
	rm -rf data/index/* .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
