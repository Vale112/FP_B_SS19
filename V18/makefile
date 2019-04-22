all:  build/main.pdf

main.pdf: build/kalibration.pdf build/Eu_log_Kanal.pdf build/tables/zuordnung_Eu.tex build/tables/Gauss-Fit-Parameter.tex build/tables/det_eff.tex build/efficiency,pdf
	python calculation/auswertung.py

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

FORCE:

.PHONY: all clean
