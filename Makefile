.PHONY: setup run test down

setup:
	docker compose up -d chromadb
	docker compose run --rm app uv run python -m scripts.setup

run:
	docker compose up -d
	docker compose run --rm app uv run python -m scripts.ask_sample_questions

test:
	docker compose run --rm app uv run pytest

down:
	docker compose down
