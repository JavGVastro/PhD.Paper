# Automatically run the jupyter notebooks

# Confidence intervals calculations
OBJECTS = Car Dor HV HX M8 N346 N595 N604H OrionS
CI_NOTEBOOKS := $(patsubst %,CI%.ipynb,$(OBJECTS))

LIBFILES=bfunc.py bplot.py

Imgs/sf-emcee-%.pdf: CI%.ipynb $(LIBFILES)
	jupyter nbconvert --to notebook --inplace --execute $< 

test:
	echo $(CI_NOTEBOOKS)
