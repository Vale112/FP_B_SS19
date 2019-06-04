all:  build/main.pdf | build

calc: data/Messdaten_1K.txt data/Messdaten_2K.txt calculation/auswertung.py | build
	python calculation/auswertung.py

build/kleineT.pdf: calc | build
build/großeT1.pdf: calc | build
build/großeT2.pdf: calc | build
build/Messdaten1.pdf: calc | build
build/Messdaten2.pdf: calc | build

build:
	mkdir -p build

build/main.pdf:


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

FORCE: build/kleineT.pdf \
build/großeT1.pdf \
build/großeT2.pdf \
build/Messdaten1.pdf \
build/Messdaten2.pdf | build

.PHONY: all clean
