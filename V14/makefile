all:  build/main.pdf

build:
	mkdir -p build

build/main.pdf: build/Nullmessung.pdf | build

build/Nullmessung.pdf: data/nullmessung calculation/auswertung.py 
	python calculation/auswertung.py


build/main.pdf: FORCE | build
	  TEXINPUTS="$(call translate,build:)" \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
		--interaction=nonstopmode
		--halt-on-error
	main.tex

clean:
	rm -rf build

FORCE:

.PHONY: all clean
