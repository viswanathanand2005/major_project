.PHONY: install install-bot install-core install-mcp test lint bot mcp demo query voice docker-build docker-run docker-run-bot clean

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[all]"

install-bot:
	pip install -e ".[bot]"

install-core:
	pip install -e .

install-mcp:
	pip install -e ".[mcp]"

# ── Run ───────────────────────────────────────────────────────────────────────
bot:
	python -m yojana_sahayak.bot.telegram_bot

mcp:
	python -m yojana_sahayak.mcp.server

query:
	@read -p "Question: " q && python -m yojana_sahayak.cli --text "$$q"

voice:
	python -m yojana_sahayak.cli --voice

demo:
	python -m yojana_sahayak.cli --gradio

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check yojana_sahayak/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t yojana-sahayak .

docker-run:
	docker run -it yojana-sahayak

docker-run-bot:
	docker run --env-file .env -p 8000:8000 yojana-sahayak python -m yojana_sahayak.bot.telegram_bot

docker-run-demo:
	docker run -p 7860:7860 yojana-sahayak python -m yojana_sahayak.cli --gradio

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info
