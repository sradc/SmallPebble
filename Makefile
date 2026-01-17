
.PHONY: test
test:
	uv run pytest

.PHONY: readme
readme:
	rm -r "README_files"  \
	&& uv run jupyter nbconvert --to markdown README.ipynb
