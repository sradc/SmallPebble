.PHONY: test readme pre-commit check-readme

test:
	uv run pytest

readme:
	rm -rf README_files
	uv run jupyter nbconvert --to markdown README.ipynb

pre-commit:
	uv run pre-commit run --all-files

check-readme: readme
	git diff --exit-code README.md
