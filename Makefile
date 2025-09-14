.PHONY: dev-extractor dev-model

dev-extractor:
	python3 catalog/extractor-pipeline/main.py

dev-model:
	python3 catalog/model-builder/test_pipeline.py