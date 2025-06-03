# WissKI Path Generator

This is a Python utility that allows the generation of a [WissKI](https://wiss-ki.eu/) pathbuilder XML file, based on a set of paths specified in a spreadsheet. Documentation of the spreadsheet input format will come at some later point. 

## Running the script

The `read_paths.py` script generates [RELEVEN](https://releven.univie.ac.at/)-style assertion paths, which are a reification of the paths specified in the spreadsheet through the [CIDOC-CRM](https://cidoc-crm.org/) `E13 Attribute Assignment` class. It can be run via uv:

    uv run src/read_paths.py -h

The required arguments are an .xlsx spreadsheet that specifies the paths to be created, and an ontology file parseable by Python's [rdflib](https://rdflib.readthedocs.io/en/stable/) to sanity-check the paths.