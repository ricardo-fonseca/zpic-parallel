.PHONY: dev build clean

dev:
	pip install --no-build-isolation -Ceditable.rebuild=true -ve .

build:
	pip wheel . -w dist/

clean:
	rm -rf build dist *.egg-info
	find python -name __pycache__ -exec rm -rf {} +
