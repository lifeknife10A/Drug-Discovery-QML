.PHONY: check test smoke

check: test smoke

test:
	pytest tests/ -v

smoke:
	python3 run.py all --smoke
