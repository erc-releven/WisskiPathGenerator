from __future__ import annotations
from collections import OrderedDict, defaultdict
from random import choice
from rdflib import Graph, RDF, OWL, URIRef
from string import hexdigits
from uuid import uuid4
from lxml import etree
from warnings import warn


class PathbuilderError(Exception):
    pass


class WisskiPath:
    """Defines the components of a path that goes into a pathbuilder."""

    enabled = 0
    group_id = 0
    disamb = 0
    weight = 0
    is_group = False
    short_name = None
    pathbuilder = None

    def __init__(self, id: str, name: str, description: str=None) -> None:
        """Create a path with the given ID and initialise the generated identifiers"""
        self.id = id
        self.name = name
        self.description = description
        # The path has a UUID; make one up
        self.uuid = uuid4()

    def enable(self) -> None:
        self.enabled = 1

    def disable(self) -> None:
        self.enabled = 0

    def generate_field_id(self) -> str:
        if self.enabled or self.is_group:
            # Generate a random 32-char hex string, substituting the first letter
            # with either 'b' or 'f' to deal with Drupal's field name restrictions
            hex_chars = hexdigits.lower()[:16]  # Get hex characters (0-9, a-f)
            wid = [choice(hex_chars) for _ in range(32)]
            wid[0] = 'b' if self.is_group else 'f'
            return ''.join(wid)
        else:
            # Return an empty-field indicator instead of doing this for real
            return 'ea6cd7a9428f121a9a042fe66de406eb'

    def set_pathspec(self, sequence: list[str], datatype_property: str='empty') -> None:
        """Given a list containing an entity / property / entity chain and an optional ending 
        data property, check that the chain elements all exist in the pathbuilder's ontology
        and then add the chain to the path."""
        if self.pathbuilder:
            # Sanity-check that all predicates and objects exist in the ontology
            g = self.pathbuilder.ontology
            # Expand the sequence and datatype to their FQ variants
            sequence = self.pathbuilder.expand_pathspec(sequence)
            check_datatype = False
            if datatype_property != 'empty':
                prefix = datatype_property.split(':', 1)[0]
                check_datatype = prefix not in ['rdf', 'rdfs', 'xsd', 'owl']
                datatype_property = self.pathbuilder.expand_pathspec([datatype_property])[0]
            # Make sure these things are actually in the ontology
            for i, e in enumerate(sequence):
                # HACK: skip the ZP78 related things as they aren't defined yet
                if 'ZP78' in e: continue
                n = URIRef(e.lstrip('^'))
                # if the index is 0, 2, 4, etc. it is a class; otherwise a property
                if i % 2 == 1 and (n, RDF.type, OWL.ObjectProperty) not in g:
                    raise PathbuilderError(f"Property {n} not found in pathbuilder ontology")
                elif i % 2 == 0 and (n, RDF.type, OWL.Class) not in g:
                    raise PathbuilderError(f"Class {n} not found in pathbuilder ontology")
            # The sequence should be an odd length, i.e. end in a class
            if len(sequence) % 2 == 0:
                raise PathbuilderError(f"Path specification should have an odd number of elements")
            # Check the datatype property too if we were asked to
            if check_datatype and (URIRef(datatype_property), RDF.type, OWL.DatatypeProperty) not in g:
                raise PathbuilderError(f"Data property {datatype_property} not found in pathbuilder ontology")
                

        # Either the sanity check passed, or this path was created outside of a pathbuilder.
        self.pathspec = sequence
        self.datatype_property = datatype_property

    def set_weight(self, weight: int) -> None:
        self.weight = weight

    def set_cardinality(self, cardinality: int | str | None) -> None:
        """Set the cardinality. Valid arguments are anything that can be cast to a positive integer, 
        the string 'Unlimited', or any false-ish value (which is interpreted as 'Unlimited')."""
        if not cardinality:
            cardinality = -1
        else:
            # See if it is a positive integer
            try:
                cardinality = int(cardinality)
                if cardinality < 1:
                    raise PathbuilderError(f"Cardinality should be 'unlimited' or a positive integer")
            except ValueError:
                if cardinality == 'Unlimited' or cardinality == 'unlimited':
                    cardinality = -1
                else:
                    raise PathbuilderError(f"Invalid cardinality setting {cardinality}; should be 'unlimited' or a positive integer")
        self.cardinality = cardinality

    def _set_supergroup(self, supergroup: WisskiPath) -> None:
        if not supergroup.is_group:
            raise PathbuilderError("The supergroup needs to be a WissKI group")
        if supergroup.pathbuilder is not self.pathbuilder:
            raise PathbuilderError("The group and supergroup need to be in the same pathbuilder")
        self.group_id = supergroup.id

    def make_group(self, pathspec: list[str], supergroup: WisskiPath, cardinality: int | str | None) -> None:
        if supergroup:
            self._set_supergroup(supergroup)
            pathspec = supergroup.pathspec + pathspec

        self.set_pathspec(pathspec)
        self.is_group = 1
        self.set_cardinality(cardinality)
        self.bundle = self.generate_field_id()
        self.field = self.bundle
        self.fieldtype = None
        self.displaywidget = None
        self.formatterwidget = None

    def make_data_path(self, pathspec: list[str], supergroup: WisskiPath, cardinality: int | str | None, valuetype: str) -> None:
        # Set the supergroup and prepend its path
        if supergroup:
            self._set_supergroup(supergroup)
            pathspec = supergroup.pathspec + pathspec
            self.bundle = supergroup.bundle
        else:
            raise PathbuilderError(f"Trying to make data path on {self.id} without any group?!")
        self.bundle = supergroup.bundle if self.enabled else None
        # Pop off the datatype property and set the path
        dp = pathspec.pop()
        self.set_pathspec(pathspec, dp)

        self.is_group = 0
        self.set_cardinality(cardinality)
        self.field = self.generate_field_id()
        # Check the datatype and set the appropriate field
        fieldargs = ('string', 'string_textfield', 'string')
        if valuetype == 'rdfs:Resource':
            fieldargs = ('uri', 'uri', 'uri_link')
        elif valuetype == 'xsd:coordinates':
            fieldargs = ('geofield', 'geofield_latlon', 'geofield_latlon')
        elif valuetype != 'xsd:string':
            warn(f'Unrecognised datatype {valuetype}; setting as string')
        self.fieldtype = fieldargs[0]
        self.displaywidget = fieldargs[1]
        self.formatterwidget = fieldargs[2]

    def make_entityref_path(self, pathspec: list[str], supergroup: WisskiPath, cardinality: int | str | None, fieldtype: str) -> None:
        if supergroup:
            self._set_supergroup(supergroup)
            pathspec = supergroup.pathspec + pathspec
            self.bundle = supergroup.bundle
        else:
            raise PathbuilderError(f"Trying to make entity reference path on {self.id} without any group?!")
        self.bundle = supergroup.bundle if self.enabled else None
        self.set_pathspec(pathspec)
        self.is_group = 0
        self.set_cardinality(cardinality)
        self.field = self.generate_field_id()
        fielddisplays = {
            'reference': 'entity_reference_autocomplete',
            'inline': 'inline_entity_form_complex'
        }
        self.fieldtype = 'entity_reference'
        self.formatterwidget = 'entity_reference_rss_category'
        self.displaywidget = fielddisplays.get(fieldtype)
        # This is set to the value of the 'x' steps in the path, which is the (number of steps + 1) / 2.
        self.disamb = int((len(pathspec) + 1) / 2)

    def to_xml(self) -> etree.Element:
        path = etree.Element("path")
        # Set the 'informative' field
        # n.b. is this necessary?
        self.field_type_informative = self.fieldtype

        # Add the first set of fields
        for field in ['id', 'weight', 'enabled', 'group_id', 'bundle', 'field', 'fieldtype', 'displaywidget', 
                      'formatterwidget', 'cardinality', 'field_type_informative']:
            e = etree.Element(field)
            if getattr(self, field) is not None:
                e.text = str(getattr(self, field))
            path.append(e)
        # Add the path
        path_array = etree.Element('path_array')
        for i, step in enumerate(self.pathspec):
            xy = 'y' if i % 2 else 'x'
            e = etree.Element(xy)
            e.text = step
            path_array.append(e)
        path.append(path_array)
        # Add the next set of fields
        for field in ['datatype_property', 'short_name', 'disamb', 'description', 'uuid', 'is_group', 'name']:
            e = etree.Element(field)
            if getattr(self, field) is not None:
                e.text = str(getattr(self, field))
            path.append(e)
        return path
    
    @staticmethod
    def from_xml(etree_element: etree.Element) -> WisskiPath:
        # Fish out the ID and name to initialise the element
        id = etree_element['id'].text
        name = etree_element['name'].text
        path = WisskiPath(id, name)
        pathspec = []
        for el in etree_element:
            if el.name == 'path_array':
                for step in el:
                    pathspec.append(step.text)
                path.set_pathspec(pathspec)
            elif el.name in ['id', 'name']:
                # We already have it
                continue
            else:
                a = el.name
                v = el.text or None
                setattr(path, a, v)
        return path


class Pathbuilder:
    """Defines a pathbuilder, which holds an ordered list of WissKI paths."""

    pathlist = OrderedDict()
    ontology = Graph()  # seeded with the standard vocabularies and no more
    default_enable = False

    def __init__(self, ontology: Graph, enable: bool=False) -> None:
        self.ontology = ontology
        self.default_enable = enable
        pass

    def add_path(self, id: str, name: str, description: str=None, enable: bool=False) -> WisskiPath:
        if id in self.pathlist:
            raise PathbuilderError(f"Path with ID {id} already exists in the pathbuilder!")
        pathobj = WisskiPath(id, name, description)
        pathobj.pathbuilder = self
        if enable or self.default_enable:
            pathobj.enable()
        self.pathlist[id] = pathobj
        return pathobj

    def delete_path(self, pathid: str):
        if id not in self.pathlist:
            warn(f"No path with ID {pathid} in this pathbuilder!")
            return
        del self.pathlist[pathid]

    def weight_paths(self):
        """Assign weights to the paths in the order they appear."""
        weights = defaultdict(int)
        for path in self.pathlist.values():
            pp = path.group_id
            path.set_weight(weights[pp])
            weights[pp] += 1  

    def expand_pathspec(self, speclist: list[str]):
        g = self.ontology
        result = []
        for p in speclist:
            # See if we have an inversion mark
            i = ''
            if p.startswith('^'):
                i = '^'
                p = p[1:]
            # Get the prefix and its expansion
            prefix, rest = p.split(':', 1)
            ns = g.namespace_manager.store.namespace(prefix)
            if ns is not None:
                # We found a fully qualified variant
                result.append(''.join([i, str(ns), rest]))
            else:
                # We didn't. Pass the original p through
                result.append(f"{i}{p}")
        return result
    
    def serialize(self):
        """Return a string which is the XML serialization of the pathbuilder."""
        self.weight_paths()
        pbtree = etree.Element('pathbuilderinterface')
        for path in self.pathlist.values():
            pbtree.append(path.to_xml())
        return etree.tostring(pbtree, pretty_print=True).decode()


