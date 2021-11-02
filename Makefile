# Automatically run the jupyter notebooks

# Confidence intervals calculations
OBJECTS = CarC Dor HV HX M8 N346 N595 N604H OrionS OrionLH2
CI_NOTEBOOKS := $(patsubst %,CI%.ipynb,$(OBJECTS))
SF_FIGS := $(patsubst %,Imgs/sf-emcee-%.pdf,$(OBJECTS))

LIBFILES=bfunc.py bplot.py

# PATTERN RULES

Imgs/sf-emcee-%.pdf: CI%.ipynb $(LIBFILES)
	jupyter nbconvert --to notebook --inplace --execute $< 

# TARGETS

sf: $(SF_FIGS)

test:
	ls -l $(SF_FIGS)
