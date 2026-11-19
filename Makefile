.PHONY: install test lint typecheck run dashboard glue-local clean

install:
	pip install -e ".[dev,spark,dashboard]"

lint:
	ruff check src tests dags glue dashboard
	ruff format --check src tests dags glue dashboard

typecheck:
	mypy

test:
	pytest -q

run:
	nqqr run all

dashboard:
	nqqr dashboard

glue-local:
	python glue/transform_job.py --local --BRONZE_PATH data/lake/bronze/bars_1m \
		--SILVER_PATH data/glue_out/silver --QUARANTINE_PATH data/glue_out/quarantine

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache
