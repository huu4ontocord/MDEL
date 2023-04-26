setup_dev:
	pip install -r requirements.txt
	pre-commit install
	pip install -e .
