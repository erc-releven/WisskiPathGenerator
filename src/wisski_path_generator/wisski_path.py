from lxml import etree
from random import choice
from string import hexdigits
from uuid import uuid4


def generate_wisski_id():
    hex_chars = hexdigits.lower()[:16]  # Get hex characters (0-9, a-f)
    return ''.join(choice(hex_chars) for _ in range(32))


class wisski_path:
    """Defines the components of a path that goes into a pathbuilder."""

    id = None
    name = 'WissKI Path'
    enabled = 0
    group_id = 0
    disamb = 0
    datatype_property = 'empty'
    field_type_informative = None
    short_name = None

    def __init__(self, id, name, description=None):
        """Create a path with the given ID and initialise the generated identifiers"""
        self.id = id
        self.name = name
        self.description = description
        self.disamb = 0
        # The path has a UUID
        self.uuid = uuid4()
        # The path has a bundle
        self.bundle = generate_wisski_id()

    def enable(self):
        self.enabled = 1

    def disable(self):
        self.enabled = 0

    def set_pathspec(self, sequence, datatype_property=None):
        """Given a list containing an entity / property / entity chain and an optional ending 
        data property, add this to the path."""
        self.pathspec = sequence
        self.datatype_property = datatype_property

    def make_group(self, pathspec, supergroup, cardinality):
        self.set_pathspec(pathspec)
        self.group_id = supergroup
        self.is_group = 1
        self.cardinality = cardinality
        self.field = self.bundle
        self.fieldtype = None
        self.displaywidget = None
        self.formatterwidget = None

    def make_data_path(self, pathspec, supergroup, cardinality, fieldtype, displaywidget, formatterwidget):
        dp = pathspec.pop()
        self.set_pathspec(pathspec, dp)
        self.group_id = supergroup
        self.is_group = 0
        self.cardinality = cardinality
        self.field = generate_wisski_id()
        self.fieldtype = fieldtype
        self.displaywidget = displaywidget
        self.formatterwidget = formatterwidget

    def make_entityref_path(self, pathspec, supergroup, cardinality):
        self.set_pathspec(pathspec)
        self.group_id = supergroup
        self.is_group = 0
        self.cardinality = cardinality
        self.field = generate_wisski_id()
        self.fieldtype = 'entity_reference'
        self.displaywidget = 'entity_reference_autocomplete'
        self.formatterwidget = 'entity_reference_rss_category'
        # n.b. is this necessary?
        self.field_type_informative = 'entity_reference'
        # This is set to the value of the 'x' steps in the path, which is the (number of steps + 1) / 2.
        self.disamb = int((len(pathspec) + 1) / 2)

    def to_xml(self):
        path = etree.Element("path")
        # Add the first set of fields
        for field in ['id', 'enabled', 'group_id', 'bundle', 'field', 'fieldtype', 'displaywidget', 'formatterwidget', 'cardinality']:
            e = etree.Element(field)
            if getattr(self, e) is not None:
                e.text = getattr(self, e)
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
            if getattr(self, e) is not None:
                e.text = getattr(self, e)
            path.append(e)
        return path
    
    @staticmethod
    def from_xml(etree_element):
        # Fish out the ID and name to initialise the element
        id = etree_element['id'].text
        name = etree_element['name'].text
        path = wisski_path(id, name)
        for el in etree_element:
            if el.name == 'path_array':
                pathspec = []
                for step in el:
                    pathspec.append(step.text)
                path.set_pathspec(pathspec)
            elif el.name in ['id', 'name']:
                continue
            else:
                a = el.name
                v = el.text or None
                setattr(path, a, v)
        return el
