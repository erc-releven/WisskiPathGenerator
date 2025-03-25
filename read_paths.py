# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lxml",
#     "openpyxl",
#     "rdflib",
# ]
# ///
import re
from collections import OrderedDict, defaultdict
from warnings import warn
from lxml import etree
from openpyxl import load_workbook
from random import choice
from rdflib import Graph
from sys import argv
from string import hexdigits, punctuation
from uuid import uuid4


class PathbuilderError(Exception):
    pass


class WisskiPath:
    """Defines the components of a path that goes into a pathbuilder."""

    enabled = 0
    group_id = 0
    disamb = 0
    is_group = False
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

    def set_cardinality(self, cardinality):
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


    def make_group(self, pathspec, supergroup, cardinality):
        if supergroup:
            if not supergroup.is_group:
                raise PathbuilderError("The supergroup needs to be a WissKI group")
            self.group_id = supergroup.id
            pathspec = supergroup.pathspec + pathspec
        self.set_pathspec(pathspec)
        self.is_group = 1
        self.set_cardinality(cardinality)
        self.field = self.bundle
        self.fieldtype = None
        self.displaywidget = None
        self.formatterwidget = None

    def make_data_path(self, pathspec, supergroup, cardinality, valuetype):
        # Set the supergroup and prepend its path
        if supergroup:
            if not supergroup.is_group:
                raise PathbuilderError("The supergroup needs to be a WissKI group")
            self.group_id = supergroup.id
            pathspec = supergroup.pathspec + pathspec
        # Pop off the datatype property and set the path
        dp = pathspec.pop()
        self.set_pathspec(pathspec, dp)

        self.is_group = 0
        self.set_cardinality(cardinality)
        self.field = self.generate_wisski_id()
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

    def make_entityref_path(self, pathspec, supergroup, cardinality):
        if supergroup:
            if not supergroup.is_group:
                raise PathbuilderError("The supergroup needs to be a WissKI group")
            self.group_id = supergroup.id
            pathspec = supergroup.pathspec + pathspec
        self.set_pathspec(pathspec)
        self.is_group = 0
        self.set_cardinality(cardinality)
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
        for field in ['id', 'weight', 'enabled', 'group_id', 'bundle', 'field', 'fieldtype', 'displaywidget', 'formatterwidget', 'cardinality']:
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
        path = WisskiPath(id, name)
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
    dpath = [get_uri(g, lineinfo['predicate'])]
    wp = WisskiPath(spid, lineinfo['label'])
    cardinality = 1 if lineinfo['type'] == 'l' else 0
    if lineinfo['entityref']:
        wp.make_entityref_path(dpath, parent, cardinality)
    else:
        wp.make_data_path(dpath, parent, cardinality, lineinfo['object'])
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
    path = []
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
            path = chain
        # Now STARify whatever the predicate was  
        chaintop, assertionclass, chainbottom = make_assertion_pathchain(g, lineinfo['predicate'])

    path.extend([chaintop, assertionclass])
    # A bare assertion has a cardinality of 1, as there should only be one of these things
    # per entity
    cardinality = 1 if lineinfo['type'] == 'b' else 'Unlimited'
    assertion = WisskiPath(id, label)
    assertion.make_group(path, parent, cardinality)
    return assertion, chainbottom


def make_object_path(g, parent, lineinfo, chainbottom):
    """Given a parent that is an assertion and the object predicate we need to
    attach something to the assertion, create the object path based on the lineinfo."""
     # The identifier line is special-cased and regular; we are making a whole object group
    # and two extra paths. Intervene here where necessary
    if lineinfo['type'] == 'identifier':
        path = [chainbottom, get_uri(g, 'crm:E42_Identifier')]
        # We need to make the object a group with two data paths
        ppp = f'{parent.id[2:]}_identifier'
        object_p = WisskiPath(f'g_{ppp}', 'External identifier')
        object_p.make_group(path, parent, 1)
        object_content = WisskiPath(object_p.id + '_plain', 'Plaintext identifier')
        object_content.make_data_path([get_uri(g, 'crm:P190_has_symbolic_content')], object_p, 1, 'xsd:string')
        object_uri = WisskiPath(object_p.id + '_is', 'URI in the database / repository')
        object_uri.make_data_path([get_uri(g, 'owl:sameAs')], object_p, 1, 'rdfs:Resource')
        # Go ahead and return it now
        return object_p, object_content, object_uri

    # In all other cases, just add the single object class    
    # Now add the main object to the path
    path = [chainbottom, get_uri(g, lineinfo['object'])]
    label = re.sub(r'^g_(.*)_assertion', r'p_\1_is', parent.id)
    object_p = WisskiPath(label, lineinfo['label'])

    # Either it's an entity reference field, or a datatype field, or a group
    # which we expect to have sub-fields
    if lineinfo['entityref']:
        object_p.make_entityref_path(path, parent, 1)
    elif len(lineinfo['remainder']):
        if len(lineinfo['remainder']) != 2:
            warn(f"Something wonky about the line specification {lineinfo}") 
        dtype_prop, dtype = lineinfo['remainder']
        path.append(get_uri(g, dtype_prop))
        object_p.make_data_path(path, parent, 1, dtype)
    elif lineinfo['type'] in 'bg':
        # It is a group that will have sub-assertions
        object_p.make_group(path, parent, 1)
    else:
        warn(f"Unable to determine object path type for {lineinfo}")

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
            label = 'p_' + parent.id[2:] + '_by'
            name = 'Database / repository'
        authority = WisskiPath(label, name)
        apath = [get_uri(g, 'crm:P14_carried_out_by'), get_uri(g, authclass)]
        authority.make_entityref_path(apath, parent, 1)
        return [authority]
    

def _make_starleg(g, parent, suffix, label, predicate, object):
    """Internal method for source and provenance legs"""
    wpid = re.sub(r'^g_(.*)_assertion', r'p_\1', parent.id) + suffix
    wp = WisskiPath(wpid, label)
    wp.make_entityref_path([predicate, object], parent, 1)
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
    # If this is a grouping for further assertions, we are done after the object
    if lineinfo['type'] in 'bg':
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
    

def weight_paths(pathlist):
    """Assign weights to the paths in the order they were written."""
    weights = defaultdict(int)
    for path in pathlist:
        pp = path.group_id
        path.set_weight(weights[pp])
        weights[pp] += 1  
    

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
        if lineinfo['type'] in 'lm':
            # It is a simple datatype path. Make it and carry on;
            # it won't be a parent to anything.
            wisski_paths.append(make_simple_path(g, parent, lineinfo))
        elif lineinfo['type'] == 'top':
            # It is a top-level group. Make it and put it on the stack
            entity = WisskiPath(make_id(lineinfo['label'], isgroup=True), lineinfo['label'])
            entity.make_group([get_uri(g, lineinfo['object'])], None, 'Unlimited')
            wisski_paths.append(entity)
            stack[0] = entity
        else:
            # It is some sort of assertion path. Make the lot
            newpaths = make_assignment_paths(g, parent, lineinfo)
            wisski_paths.extend(newpaths)
            # ...and add the object as the parent at this level
            stack[lineinfo['level']] = newpaths[1]
    # Add the path weights all in one go
    weight_paths(wisski_paths)

    # Now serialise all those paths
    pathbuilder = etree.Element('pathbuilderinterface')
    for path in wisski_paths:
        pathbuilder.append(path.to_xml())
    print(etree.tostring(pathbuilder, pretty_print=True).decode())