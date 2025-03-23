# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lxml",
#     "openpyxl",
#     "rdflib",
# ]
# ///
import re
from collections import OrderedDict
from warnings import warn
from lxml import etree
from openpyxl import load_workbook
from random import choice
from rdflib import Graph
from sys import argv
from string import hexdigits, punctuation
from uuid import uuid4


class wisski_path:
    """Defines the components of a path that goes into a pathbuilder."""

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
        self.weight = 0
        self.description = description
        self.disamb = 0
        # The path has a UUID
        self.uuid = uuid4()
        # The path has a bundle
        self.bundle = self.generate_wisski_id()

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
        if supergroup:
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
        if supergroup:
            self.group_id = supergroup
        self.is_group = 0
        self.cardinality = cardinality
        self.field = self.generate_wisski_id()
        self.fieldtype = fieldtype
        self.displaywidget = displaywidget
        self.formatterwidget = formatterwidget

    def make_entityref_path(self, pathspec, supergroup, cardinality):
        self.set_pathspec(pathspec)
        if supergroup:
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


def get_star_assertion(predicate):
    prefix, name = predicate.split(':')
    pid = name.split('_')[0]
    return f'star:E13_{prefix}_{pid}'


def make_id(descname, isgroup=False):
    l = 'g_' if isgroup else 'p_'
    clean_name = [x for x in descname.lower() if x not in punctuation]
    return l + ''.join(clean_name).replace(' ', '_')


def make_simple_path(g, parent, lineinfo):
    """Takes a parent group, e.g. Boulloterion, and creates a simple datatype path
    based on the lineinfo, e.g. for rdf:label. Returns the single path"""
    # e.g. p_boulloterion_descriptive_name
    spid = make_id(" ".join([parent.name, lineinfo['label']]))
    dpath = parent.pathspec.copy()
    dpath.extend([get_uri(g, lineinfo['predicate']), get_uri(g, lineinfo['object'])])
    wp = wisski_path(spid, lineinfo['label'])
    # TODO check whether the object is xsd:string or something else
    wp.make_data_path(dpath, parent.id, 1, 'string', 'string_textfield', 'string')
    return wp


def make_assertion_pathchain(g, predicate):
    """Takes a predicate/object pair and returns the correct STARified path chain"""
    chaintop = 'crm:P140_assigned_attribute_to'
    chainbottom = 'crm:P141_assigned'
    assertionclass = get_star_assertion(predicate.lstrip('^'))
    if lineinfo['predicate'].startswith('^'):
        # The object, not the subject, is on top
        chainbottom, chaintop = chaintop, chainbottom
    # FQ the lot and reverse the top of the chain
    return '^' + get_uri(g, chaintop), get_uri(g, assertionclass), get_uri(g, chainbottom)


def make_assertion_path(g, parent, lineinfo):
    """Takes a parent group that is an object and returns the assertion group on this
    parent, based on the lineinfo."""
    # First make the assignment group for the assertion.     
    # We need to get the right STAR assertion and predicates for this
    chaintop = 'crm:P140_assigned_attribute_to'
    chainbottom = 'crm:P141_assigned'
    label = lineinfo['label'] + ' assertion' if lineinfo['label'] else ''
    id = make_id(f"{parent.name} {label}", isgroup=True)
    path = parent.pathspec.copy()
    if lineinfo['type'] == 't':
        # The assertion is an E17 with its specialised predicates
        assertionclass = 'crm:E17_Type_Assignment'
        chaintop = 'crm:P41_classified'
        chainbottom = 'crm:P42_assigned'
    elif lineinfo['type'] == 'identifier':
        # The assertion is an E15 with its specialised object predicate
        assertionclass = 'crm:E15_Identifier_Assignment'
        chainbottom = 'crm:P37_assigned'
        # The labels are standard strings
        label = 'Identity in other services'
        id = parent.id + '_id_assignment'
    else:
        # Extend the base chain if this is a compound path
        if lineinfo['type'] == 'c':
            # Compound paths have a path infix, i.e. extra elements, of which the predicates still
            # need to be starified.
            # Make the predicate / object / predicate / object chain as deep as we need to.
            chain = []
            while len(lineinfo['prefix']):
                predicate = lineinfo['prefix'].pop(0)
                object = lineinfo['prefix'].pop(0)
                chain.extend(make_assertion_pathchain(g, predicate))
                chain.append(get_uri(g, object))
            path.extend(chain)
        # Now STARify whatever the predicate was  
        chaintop, assertionclass, chainbottom = make_assertion_pathchain(g, lineinfo['predicate'])

    path.extend([chaintop, assertionclass])
    assertion = wisski_path(id, label)
    assertion.make_group(path, parent.id, -1)
    return assertion, chainbottom


def make_object_path(g, parent, lineinfo, chainbottom):
    """Given a parent that is an assertion and the object predicate we need to
    attach something to the assertion, create the object path based on the lineinfo."""
     # The identifier line is special-cased and regular; we are making a whole object group
    # and two extra paths. Intervene here where necessary
    if lineinfo['type'] == 'identifier':
        path = parent.pathspec + [chainbottom, get_uri(g, 'crm:E42_Identifier')]
        # We need to make the object a group with two data paths
        ppp = f'{parent.id[2:]}_identifier'
        object_p = wisski_path(f'g_{ppp}_identifier', 'External identifier')
        object_p.make_group(path, parent.id, 1)
        object_content = wisski_path(object_p.id + '_plain', 'Plaintext identifier')
        object_content.make_data_path(path + [get_uri(g, 'crm:P190_has_symbolic_content'), get_uri(g, 'xsd:string')],
                                       object_p.id, 1, 'string', 'string_textfield', 'string')
        object_uri = wisski_path(object_p.id + '_is', 'URI in the database / repository')
        object_uri.make_data_path(path + [get_uri(g, 'owl:sameAs'), get_uri(g, 'crm:E42_Identifier')],
                                   object_p.id, 1, 'uri', 'uri', 'uri_link')
        # Go ahead and return it now
        return object_p, object_content, object_uri

    # In all other cases, just add the single object class    
    path = parent.pathspec.copy()
    # Now add the main object to the path
    path.append(chainbottom)
    label = re.sub(r'^g_(.*)_assertion', r'p_\1_is', parent.id)
    object_p = wisski_path(label, lineinfo['label'])

    # and either the entity reference field, or the datatype field
    if lineinfo['entityref']:
        object_p.make_entityref_path(path, parent, 1)
    else:
        path += [get_uri(g, x) for x in lineinfo['remainder']]
        # TODO check the actual datatype
        object_p.make_data_path(path, parent, 1, 'string', 'string_textfield', 'string')
    return object_p


def make_authority_paths(g, parent, authclass):
    """Takes a parent group that is an assertion, e.g. star:E13_crm_P108 or 
    crm:E15_Identifier_Assignment. Returns either one or two STAR leg paths for the
    authority, depending on what class(es) the authority should be."""
    if authclass == 'crm:E39_Actor':
        # This is the one that needs two paths
        return [make_authority_paths(g, parent, 'crm:E21_Person')[0], 
                make_authority_paths(g, parent, 'spec:Author_Group')[0]]
    else:
        # Here is the 'real' logic
        label = re.sub(r'^g_(.*)_assertion', r'p_\1_by', parent.id)
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
    

def _make_starleg(g, parent, suffix, label, predicate, object):
    """Internal method for source and provenance legs"""
    wpid = re.sub(r'^g_(.*)_assertion', r'p_\1', parent.id) + suffix
    wp = wisski_path(wpid, label)
    wpath = parent.pathspec.copy()
    wpath.extend([predicate, object])
    wp.make_entityref_path(wpath, parent, 1)
    return wp


def make_source_leg(g, parent):
    """Takes a parent group that is an assertion and returns its source"""
    return _make_starleg(g, parent, '_src', 'Found in', 
                         '^' + get_uri(g, 'crm:P67_refers_to'), get_uri(g, 'spec:Passage'))


def make_provenance_leg(g, parent):
    """Takes a parent group that is an assertion and returns its provenance"""
    return _make_starleg(g, parent, '_based', 'Based on', 
                         get_uri(g, 'crm:P17_was_motivated_by'), get_uri(g, 'spec:Bibliography'))


def make_assignment_paths(g, parent, lineinfo):
    """Takes a parent group that is an object, e.g. crm:E21_Person or crm:E67_Birth.
    Returns the assertion and its appropriate STAR legs based on the line info for
    object, authority, source, and provenance."""
    # Get the assertion group itself
    assertion, chainbottom = make_assertion_path(g, parent, lineinfo)
    # Then the path for the object
    object_p = make_object_path(g, assertion, lineinfo, chainbottom)
    # If the assertion type is bare, then we are done
    if lineinfo['type'] == 'b':
        return assertion, object_p 

    # Then the path for the authority
    if lineinfo['type'] == 'identifier':
        authorities = make_authority_paths(g, assertion, 'lrmoo:F11_Corporate_Body')
        # Return the lot - two objects, one authority, no sources
        assignment = [assertion]
        assignment.extend(object_p)
        assignment.extend(authorities)
        return assignment
    else:
        authorities = make_authority_paths(g, assertion, 'crm:E39_Actor')

    # Finally, the paths for source and provenance
    src_p = make_source_leg(g, assertion)
    prov_p = make_provenance_leg(g, assertion)

    # Return the lot
    assignment = [assertion, object_p]
    assignment.extend(authorities)
    assignment.extend([src_p, prov_p])
    return assignment


def parse_line(line):
    # Return a hash with the line's level, type, label, main predicate, 
    # main object, and remaining path parts
    result = {
        'level': 0,
        'type': None,
        'label': None,
        'predicate': None,
        'object': None,
        'entityref': False,
        'remainder': []
    }
    # Find the first component and see how deep it is
    idx = -1
    first = None
    while not first:
        idx += 1
        try:
            first = line[idx]
        except IndexError:
            break
    if first is None:
        # This was an empty line
        return None
    result['level'] = idx
    if first == 'identifier':
        # This is an identifier line; no more values are filled in
        result['type'] = first
        return result
    if idx == 0:
        # This is a top-level group line
        result.update({'type': 'top', 'label': first, 'object': line[1]})
        return result
    
    # Otherwise, this is a typed label
    result['type'] = first[0]
    result['label'] = first[2:]
    if result['type'] == 'c':
        # If it is a compound path, we need to do more below. Save the whole chain for now
        remainder = [x for x in line[idx+1:] if x is not None]
    else:
        # If it isn't a compound path, the predicate and object are at positions 1 and 2
        result['predicate'] = line[idx+1]
        result['object'] = line[idx+2]
        # and the rest is either 'reference' or a datatype pair.
        remainder = [x for x in line[idx+3:] if x is not None]

    if len(remainder):
        # Is it an entity reference? Mark it as such if so
        offset = 2
        if remainder[-1] == 'reference':
            result['entityref'] = True
            remainder.pop()
            offset = 0
        # Is it a compound line? We have to pick out the predicate and object, and set a prefix
        if result['type'] == 'c':
            # The real predicate and object are the last two items if this was an entity reference,
            # otherwise they are the second-to-last pair
            cutpoint = len(remainder) - offset - 2
            result['predicate'] = remainder[cutpoint]
            result['object'] = remainder[cutpoint+1]
            result['prefix'] = remainder[:cutpoint]
            remainder = remainder[cutpoint+2:]
        # If not, we have already set the predicate and object, and the remainder gets saved as is.
        elif len(remainder) > 2:
            warn(f"Line {line} has a suspiciously long chain; is it compound?")
        result['remainder'] = remainder
    return result


def _top(stack, pop=False):
    """Utility function to pop off the end of the stack (if requested) and return what is 
    the new top. If the stack is or becomes empty, return None"""
    if pop:
        try:
            stack.popitem()
        except KeyError:
            return None
    
    return list(stack.items())[-1] if len(stack) else None



def find_parent(stack, lineinfo):
    """Returns the parent object for the given line based on the stack. Pops things off
    the stack if necessary."""
    if len(stack) == 0:
        # There is no parent
        return None
    
    level = int(lineinfo['level'])
    if lineinfo['type'] == 'top' or level == 0:
        # Reset the stack
        stack = OrderedDict()
        return None
    top = _top(stack)  # If we've got here, the stack isn't empty

    # If the current item is a lower level than the top of the stack, then keep
    # popping things off. If we pop off the whole stack, there is no parent
    while level < top[0]:
        top = _top(stack, pop=True)
        # This shouldn't happen, but just in case
        if top is None:
            return None
    # Now the level is either equal or (was always) more than the top of the stack
    if level > top[0]:
        # The top of the stack is the parent
        return top[1]
    elif level == top[0]:
        # The top of the stack is no longer parent to anything.
        # Get rid of it and return the new top
        return _top(stack, pop=True)[1] if len(stack) else None
    

if __name__ == '__main__':
    FN = argv[1] if len(argv) > 1 else 'data/wisski_canonical_paths.xlsx'
    ONT = argv[2] if len(argv) > 2 else '../ontologies/releven-star.ttl'
    pathspec = load_workbook(filename=FN)['paths']
    g = Graph()
    g.parse(ONT)

    wisski_paths = []
    # Work through the input line by line, creating the necessary paths
    stack = OrderedDict()
    first_skipped = False
    for row in pathspec.values:
        if not first_skipped:
            first_skipped = True
            continue
        lineinfo = parse_line(row)
        parent = find_parent(stack, lineinfo)
        if lineinfo['type'] == 'l':
            # It is a simple datatype path. Make it and carry on;
            # it won't be a parent to anything.
            wisski_paths.append(make_simple_path(g, parent, lineinfo))
        elif lineinfo['type'] == 'top':
            # It is a top-level group. Make it and put it on the stack
            entity = wisski_path(make_id(lineinfo['label'], isgroup=True), lineinfo['label'])
            entity.make_group([get_uri(g, lineinfo['object'])], None, -1)
            wisski_paths.append(entity)
            stack[0] = entity
        else:
            # It is some sort of assertion path. Make the lot
            newpaths = make_assignment_paths(g, parent, lineinfo)
            wisski_paths.extend(newpaths)
            # ...and add the object as the parent at this level
            stack[lineinfo['level']] = newpaths[1]

    # Now serialise all those paths
    pathbuilder = etree.Element('pathbuilderinterface')
    for path in wisski_paths:
        pathbuilder.append(path.to_xml())
    print(etree.tostring(pathbuilder, pretty_print=True).decode())