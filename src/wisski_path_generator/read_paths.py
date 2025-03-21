from lxml import etree
from openpyxl import load_workbook
from random import choice
from rdflib import Graph, URIRef, RDF, OWL
from sys import argv
from string import hexdigits
from uuid import uuid4


class wisski_path:
    """Defines the components of a path that goes into a pathbuilder."""

    enabled = 0
    group_id = 0
    disamb = 0
    datatype_property = 'empty'
    field_type_informative = None
    short_name = None

    def __init__(self, id, name, weight, description=None):
        """Create a path with the given ID and initialise the generated identifiers"""
        self.id = id
        self.name = name
        self.weight = 0
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

    def set_weight(self, weight):
        self.weight = weight

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
        self.field = self.generate_wisski_id()
        self.fieldtype = fieldtype
        self.displaywidget = displaywidget
        self.formatterwidget = formatterwidget

    def make_entityref_path(self, pathspec, supergroup, cardinality):
        self.set_pathspec(pathspec)
        self.group_id = supergroup
        self.is_group = 0
        self.cardinality = cardinality
        self.field = self.generate_wisski_id()
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
    
    @staticmethod
    def generate_wisski_id():
        hex_chars = hexdigits.lower()[:16]  # Get hex characters (0-9, a-f)
        return ''.join(choice(hex_chars) for _ in range(32))


def get_uri(graph, ustr):
    nslist = {x: y for x, y in graph.namespace_manager.namespaces()}
    prefix, rest = ustr.split(':')
    return str(nslist[prefix]) + rest


def make_id(descname, isgroup=False):
    l = 'g_' if isgroup else 'p_'
    return l + descname.lower().replace(' ', '_')


def make_simple_path(g, parent, label, property, datatype):
    spid = make_id(parent.name + " descname")
    spname = f"{parent.name} descriptive name"
    wp = wisski_path(spid, spname)
    dpath = parent.pathspec.copy()
    dpath.extend([get_uri(g, property), get_uri(g, datatype)])
    wp.make_data_path(dpath, parent.id, 1, 'string', 'string_textfield', 'string')
    return wp


def make_authority_paths(g, parent, authclass):
    # Return the authority path(s) for the given parent.
    if authclass == 'crm:E39_Actor':
        return [make_authority_paths(g, parent, 'crm:E21_Person'), make_authority_paths(g, parent, 'spec:Author_Group')]
    else:
        label = f"p_{make_id(parent.name)}_by"
        name = 'According to'
        if 'group' in authclass.lower():
            label += '_group'
            name += ' (group)'
        elif 'F11' in authclass:
            name = 'Database / repository'
        authority = wisski_path(label, name)
        apath = parent.pathspec.copy()
        apath.extend([get_uri(g, 'crm:P14_carried_out_by'), get_uri(g, authclass)])
        authority.make_entityref_path(apath, parent.id, 1)
        return [authority]
    

def make_assignment_paths(g, parent, eclass, labelpart, gname, subjpred, authclass):
    wisski_paths = []
    ppp = parent.id[2:]  # get rid of the g_ prefix
    # First make the assignment group for the assertion
    assertion = wisski_path(f"g_{ppp}_{labelpart}_assignment", gname)
    path = parent.pathspec
    path.append(f"^{subjpred}")
    path.append(get_uri(g, eclass))
    assertion.make_group(path, parent.id, -1)
    wisski_paths.append(assertion)

    # Then the path for the authority
    authclass = 'lrmoo:F11_Corporate_Body' if 'E15_' in eclass else 'crm:E39_Actor'
    wisski_paths.extend(make_authority_paths(g, assertion, authclass))

    # If it's an identifier assignment then we are done
    if 'E15_' in eclass:
        return wisski_paths
    
    # Otherwise we need to add the source / provenance legs
    srcwp = wisski_path(f"p_{ppp}_source")
    srcwp.make_entityref_path(
        path.copy().extend(['^' + get_uri(g, 'crm:P67_refers_to'), 
                            get_uri(g, 'spec:Passage')]), 
        parent, 1)
    wisski_paths.append(srcwp)
    provwp = wisski_path(f"p_{ppp}_based")
    provwp.make_entityref_path(
        path.copy().extend([get_uri(g, 'crm:P17_was_motivated_by'), 
                            get_uri(g, 'spec:Bibliography')]), 
        parent, 1)

    # Return the wisski objects so that the 'meat' path can now be added
    return wisski_paths.extend([srcwp, provwp])


def make_identifier_paths(g, parent):
    # Set up the E13 subclass stuff
    wpaths = make_assignment_paths(g, parent, 'crm:E15_Identifier_Assignment', 'identifier', 'Identity in other services', 
                                   'crm:P140_assigned_attribute_to', 'lrmoo:F11_Corporate_Body')
    # Set up the object
    identifier = wisski_path(f"{parent.id}_identifier_is", "External identifier")
    ipath = parent.pathspec.copy().extend([get_uri(g, 'crm:P37_assigned'), get_uri(g, 'crm:E42_Identifier')])
    identifier.make_group(ipath, parent, -1)
    content = wisski_path(f"p_{parent.id}_identifier_plain", "Identifier content")  # fix the label
    content.make_data_path(ipath + get_uri(g, 'crm:P190_has_symbolic_content'), identifier.id, 1, 'string', 'string_textfield', 'string')
    sameas = wisski_path(f"p_{parent.id}_identifier_uri", "Identifier URI")
    sameas.make_data_path(ipath + get_uri(g, 'owl:sameAs'), identifier.id, 1, 'uri', 'uri', 'uri_link')
    wpaths.extend([identifier, content, sameas])
    return wpaths

def make_star_paths(g, parent, elements):
    # Starting from the parent path, we need to create the necessary STAR legs
    # with the P140 and P141 in the correct direction.
    # First we make the group for the assertion
    chaintop = get_uri(g, 'crm:P140_assigned_attribute_to')
    chainbottom = get_uri(g, 'crm:P141_assigned')
    if elements[0].startswith('^'):


def starify_line(g, line, parent=None):
    # Parse the first cell in the line to get the label and the type of pathbuilding.
    # Base all built paths on the wisski_path given in `parent`.
    path = parent.pathspec if parent is not None else []
    wisski_elements = []
    pathlabel = line.pop(0)
    if pathlabel.startswith('l '):
        # It is a simple datatype property path
        return [make_simple_path(g, parent, pathlabel[2:], line[0], line[1]))]
    if pathlabel == 'identifier':
        return make_identifier_paths(g, parent)
    if pathlabel.startswith('b '):
        # It is a fully STARified path
        for element in line:
            if not element:
                # skip blank cells
                continue 
            if element == 'reference':
                # We've reached the end
                return path, element
            # It might be an inverse property
            inverse = element.startswith('^')
            element = element.lstrip('^')
            # Check whether the element is a property at all
            if ((URIRef(get_uri(element)), RDF.type, ))


def ingest_group(g, pathbuilder, lines, supergroup=None):
    """Ingest a grouped hierarchy of path lines and add them to the pathbuilder."""
    leader = lines.pop(0)
    group_name = label(leader[0])
    # Create the group
    topgroup = wisski_path(group_name, leader[0])
    topgroup.make_group(leader[1], supergroup, -1)
    subgroup = []
    for line in lines:
        # Remove the first N empty cells
        if supergroup is None:
            line = line[2:]
        else:
            line = line[3:]
        # See if we need to create a subgroup
        if line[-1] == 'reference':
            pass


if __name__ == '__main__':
    pathspec = load_workbook(filename=argv[1])['paths']
    g = Graph()
    g.parse(argv[2])
    nm = g.namespace_manager
    