# WissKI Path Generator

This is a Python utility that allows the generation of a [WissKI](https://wiss-ki.eu/) pathbuilder XML file, based on a set of paths specified in a spreadsheet. The point of this script is to avoid the pain of UI configuration on Drupal. The included script specifically generates [RELEVEN](https://releven.univie.ac.at/)-style assertion paths; a script that generates 'vanilla' paths could easily be assembled from the `wisski_pathbuilder.Pathbuilder` module and the line parsing function in `src/read_paths.py`.

## Excel file format

The script expects an Excel file that lists paths in a hierarchical fashion, with column headings as follows:

| Label | Entity | Line type | Label | Property | Entity | Label | Property | Entity | ... |
|-------|--------|-----------|-------|----------|--------|-------|----------|--------|-----|
| Person|crm:E21_Person|
| | | l | Display name | rdfs:label | xsd:string |
| | | g | Birth event  | ^crm:P98_brought_into_life | crm:E67_Birth |
| | | l | | | | Date of birth | crm:P4_has_time-span | crm:E52_Time-Span | 

The two leftmost columns are filled out for top-level groups (which are assumed to be groups), and the path type is filled out for all child groups. A line (== path) should have only a single label, and the label should be the first piece of information after the line type; the path hierarchy is indicated by the column offset of the label respective to the line above. The spreadsheet can have as many columns as is necessary to represent the hierarchy.

## Special property values

If the property value is one of the following strings, then the generated path will be of type "Entity reference", where the referenced entity is the right-most entity in the path specificaion. The type of entity reference will be set as follows:

- `reference`: `entity_reference_autocomplete`
- `inline`: `inline_entity_form_complex`


## Line types

The following line types are supported:

- l : a regular path that ends in a datatype
- m : a regular path that ends in an entity reference
- g : a group under which some paths will be clustered

For RELEVEN-style STAR assertions, the following additional line types are supported

- a : a normal assertion (predicate-specific subclass of `crm:E13_Attribute_Assignment`)
- b : a "bare" assertion, where the source and authority legs are not generated
- c : a compound assertion, which has multiple steps. Source and authority legs are generated only on the last step.
- t : a `crm:E17_Type_Assignment` assertion
- i : a `crm:E15_Identifier_Assignment` assertion. This line will also generate a standard label ("External Identifier") and the child paths `-> crm:P37_assigned : crm:E42_Identifier` and `-> crm:P14_carried_out_by : lrmoo:F11_Corporate_Body`. 

## Running the script

The `read_paths.py` script generates [RELEVEN](https://releven.univie.ac.at/)-style assertion paths, which are a reification of the paths specified in the spreadsheet through the [CIDOC-CRM](https://cidoc-crm.org/) `E13 Attribute Assignment` class. It can be run via uv, as follows (this will print a help message with the available script flags):

    uv run src/read_paths.py -h

The required arguments are an .xlsx spreadsheet that specifies the paths to be created, and an ontology file parseable by Python's [rdflib](https://rdflib.readthedocs.io/en/stable/) to sanity-check the paths.