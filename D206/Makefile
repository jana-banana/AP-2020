all: build/main.pdf

# hier Python-Skripte:
build/tempK.pdf: AUS1.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python AUS1.py

build/TempDruck.pdf: Massendurchsatz.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python Massendurchsatz.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/tempK.pdf 
				build/TempDruck.pdf
				build/Darstellung_Waermepumpe.pdf
				build/Aufbau_Waermepumpe.pdf

build/main.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
