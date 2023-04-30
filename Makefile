VENV           = .venv
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))

$(VENV_PYTHON):
	if [ ! -d "$(VENV)" ]; then $(SYSTEM_PYTHON) -m venv $(VENV); \
	else \
		echo "Virtual environment already exists in directory $(VENV)"; \
	fi

venv: $(VENV_PYTHON)

setup_dev:
	pip install -r requirements.txt
	pre-commit install
	pip install -e .
