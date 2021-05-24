all: build/main.pdf

# hier Python-Skripte:
build/stat_plot.pdf: plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python plot.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/stat_plot.pdf
				build/plot_diff.pdf
				build/plot_messing.pdf
				build/plot_aluminium.pdf
				build/plot_edelstahl.pdf
				platine.pdf


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
