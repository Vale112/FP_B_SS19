all:  build/main.pdf

build/main.pdf: build/kalibration.pdf build/Eu_log_Kanal.pdf build/tables/zuordnung_Eu.tex build/tables/Gauss-Fit-Parameter.tex build/tables/det_eff.tex build/efficiency.pdf build/Cs_log.pdf build/vollpeak.pdf build/tables/zuordnung_Ba.tex build/tables/aktivitaet_ba.tex build/tables/Ba.tex build/mystery1_log.pdf build/Uran.pdf

build/kalibration.pdf: data/Eu152.txt calculation/auswertung.py
	python calculation/auswertung.py

build/Eu_log_Kanal.pdf: data/Eu152.txt calculation/auswertung.py
	python calculation/auswertung.py

build/tables/zuordnung_Eu.tex: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt
	python calculation/auswertung.py

build/tables/Gauss-Fit-Parameter.tex: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt
	python calculation/auswertung.py

build/tables/det_eff.tex: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt
	python calculation/auswertung.py

build/efficiency.pdf: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt
	python calculation/auswertung.py

build/Cs_log.pdf: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/Cs137.txt
	python calculation/auswertung.py

build/vollpeak.pdf: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/Cs137.txt
	python calculation/auswertung.py

build/mystery1_log.pdf: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/mystery1.txt
	python calculation/auswertung.py

build/tables/zuordnung_Ba.tex: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/mystery1.txt
	python calculation/auswertung.py

build/tables/Ba.tex: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/mystery1.txt
	python calculation/auswertung.py

build/tables/aktivitaet_ba.tex: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/mystery1.txt
	python calculation/auswertung.py

build/Uran.pdf: data/Eu152.txt calculation/auswertung.py data/2_0/Eu.txt data/salz.txt
	python calculation/auswertung.py

build:
	mkdir -p build


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
