# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lxml",
#     "openpyxl",
#     "rdflib",
# ]
# ///
import re
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from warnings import warn
from lxml import etree
from openpyxl import load_workbook
from random import choice
from rdflib import Graph, URIRef, RDF, OWL
from sys import argv, stderr
from string import hexdigits, punctuation
from uuid import uuid4


class PathbuilderError(Exception):
    pass


class Pathbuilder:
    """Defines a pathbuilder, which holds an ordered list of WissKI paths."""

    pathlist = OrderedDict()
    ontology = Graph()  # seeded with the standard vocabularies and no more
    default_enable = False

    def __init__(self, ontology, enable=False):
        self.ontology = ontology
        self.default_enable = enable
        pass

    def add_path(self, id, name, description=None, enable=False):
        if id in self.pathlist:
            raise PathbuilderError(f"Path with ID {id} already exists in the pathbuilder!")
        pathobj = WisskiPath(id, name, description)
        pathobj.pathbuilder = self
        if enable or self.default_enable:
            pathobj.enable()
        self.pathlist[id] = pathobj
        return pathobj

    def delete_path(self, pathid):
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

    def expand_pathspec(self, speclist):
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


class WisskiPath:
    """Defines the components of a path that goes into a pathbuilder."""

    enabled = 0
    group_id = 0
    disamb = 0
    weight = 0
    is_group = False
    short_name = None
    pathbuilder = None

    def __init__(self, id, name, description=None):
        """Create a path with the given ID and initialise the generated identifiers"""
        self.id = id
        self.name = name
        self.description = description
        # The path has a UUID; make one up
        self.uuid = uuid4()

    def enable(self):
        self.enabled = 1

    def disable(self):
        self.enabled = 0

    def generate_field_id(self):
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

    def set_pathspec(self, sequence, datatype_property='empty'):
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

    def set_weight(self, weight):
        self.weight = weight

    def set_cardinality(self, cardinality):
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

    def _set_supergroup(self, supergroup):
        if not supergroup.is_group:
            raise PathbuilderError("The supergroup needs to be a WissKI group")
        if supergroup.pathbuilder is not self.pathbuilder:
            raise PathbuilderError("The group and supergroup need to be in the same pathbuilder")
        self.group_id = supergroup.id

    def make_group(self, pathspec, supergroup, cardinality):
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

    def make_data_path(self, pathspec, supergroup, cardinality, valuetype):
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

    def make_entityref_path(self, pathspec, supergroup, cardinality, fieldtype):
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

    def to_xml(self):
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
    def from_xml(etree_element):
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
        return el


class STARPathMaker:
    def __init__(self, pathbuilder, expand_actors=False, no_external=False):
        self.pathbuilder = pathbuilder
        self.expand_actors = expand_actors
        self.no_external = no_external

    @staticmethod
    def get_star_entities(predicate):
        reversed = predicate.startswith('^')
        prefix, name = predicate.lstrip('^').split(':')
        pid = name.split('_')[0]
        s, o = ('P141', 'P140') if reversed else ('P140', 'P141') 
        return f'star:{s}_{prefix}_{pid}', f'star:E13_{prefix}_{pid}', f'star:{o}_{prefix}_{pid}'

    @staticmethod
    def make_id(descname, isgroup=False):
        l = 'g_' if isgroup else 'p_'
        clean_name = [x for x in descname.lower() if x not in punctuation]
        return l + ''.join(clean_name).replace(' ', '_')

    def make_simple_path(self, parent, lineinfo):
        """Takes a parent group, e.g. Boulloterion, and creates a simple datatype path
        based on the lineinfo, e.g. for rdf:label. Returns the single path"""
        # e.g. p_boulloterion_descriptive_name
        spid = self.make_id(" ".join([parent.name, lineinfo['label']]))
        dpath = [lineinfo['predicate']]
        wp = self.pathbuilder.add_path(spid, lineinfo['label'])
        cardinality = 1 if lineinfo['type'] == 'l' else 0
        if lineinfo['entityref']:
            dpath.append(lineinfo['object'])
            wp.make_entityref_path(dpath, parent, cardinality, lineinfo['entityref'])
        else:
            wp.make_data_path(dpath, parent, cardinality, lineinfo['object'])
        return wp

    def make_assertion_pathchain(self, predicate):
        """Takes a predicate/object pair and returns the correct STARified path chain"""
        chaintop, assertionclass, chainbottom = self.get_star_entities(predicate)
        # FQ the lot and reverse the top of the chain
        return '^' + chaintop, assertionclass, chainbottom

    def make_assertion_path(self, parent, lineinfo):
        """Takes a parent group that is an object and returns the assertion group on this
        parent, based on the lineinfo."""
        # Accrete the label
        label = lineinfo['label'] + ' assertion' if lineinfo['label'] else ''
        # This makes something like g_time_frame_assertion. Prepending the parent name as a 
        # seed to make_id would make something like g_creation_time_frame_assertion.
        # But we actually want g_publication_creation_time_frame_assertion. Here we rely on
        # their both starting with g_
        id = parent.id + self.make_id(label, isgroup=True)[1:]

        # Now make the assertion (E13 subclass) group for the assignment.     
        # We need to get the right STAR assertion class and predicates for this
        path = []
        if lineinfo['type'] == 't':
            # The assertion is an E17 with its specialised predicates
            assertionclass = 'crm:E17_Type_Assignment'
            chaintop = '^crm:P41_classified'
            chainbottom = 'crm:P42_assigned'
        elif lineinfo['type'] == 'identifier':
            # The assertion is an E15 with its specialised object predicate
            assertionclass = 'crm:E15_Identifier_Assignment'
            chaintop = '^crm:P140_assigned_attribute_to'
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
                    chain.extend(self.make_assertion_pathchain(predicate))
                    chain.append(object)
                path = chain
            # Now STARify whatever the predicate was  
            chaintop, assertionclass, chainbottom = self.make_assertion_pathchain(lineinfo['predicate'])

        path.extend([chaintop, assertionclass])
        # A bare assertion has a cardinality of 1, as there should only be one of these things
        # per entity
        assertion = self.pathbuilder.add_path(id, label)
        assertion.make_group(path, parent, lineinfo['type'] == 'b')
        return assertion, chainbottom


    def make_object_legs(self, parent, lineinfo, chainbottom):
        """Given a parent that is an assertion and the object predicate we need to
        attach something to the assertion, create the object path based on the lineinfo."""
        # The identifier line is special-cased and regular; we are making a whole object group
        # and two extra paths. Intervene here where necessary
        if lineinfo['type'] == 'identifier':
            path = [chainbottom, 'crm:E42_Identifier']
            # We need to make the object a group with two data paths
            ppp = f'{parent.id[2:]}_identifier'
            object_p = self.pathbuilder.add_path(f'g_{ppp}', 'External identifier')
            object_p.make_group(path, parent, 1)
            object_content = self.pathbuilder.add_path(f'p_{ppp}_plain', 'Plaintext identifier')
            object_content.make_data_path(['crm:P190_has_symbolic_content'], object_p, 1, 'xsd:string')
            if self.no_external:
                # We can't have the link to the external URI in WissKI, so just return what we have
                return [object_p, object_content]
            object_uri = self.pathbuilder.add_path(f'p_{ppp}_is', 'URI in the database / repository')
            object_uri.make_data_path(['owl:sameAs'], object_p, 1, 'rdfs:Resource')
            # Go ahead and return it all now
            return [object_p, object_content, object_uri]

        # Are we expanding E39s?
        created_paths = []
        pathobjs = [lineinfo['object']]
        if self.expand_actors and lineinfo['object'] == 'crm:E39_Actor':
            pathobjs = ['crm:E21_Person', 'crm:E74_Group']
        # Now for each object we have to deal with, make an object path
        for i, obj in enumerate(pathobjs):
            path = [chainbottom, obj]
            lreplace = r'p_\1_are' if i else r'p_\1_is'
            label = re.sub(r'^g_(.*)_assertion', lreplace, parent.id)
            object_p = self.pathbuilder.add_path(label, lineinfo['label'])

            # Either it's an entity reference field, or a datatype field, or a group
            # which we expect to have sub-fields
            if lineinfo['entityref']:
                object_p.make_entityref_path(path, parent, 1, lineinfo['entityref'])
            elif len(lineinfo['remainder']):
                # There should be a datatype property and a literal type in the remainder
                if len(lineinfo['remainder']) != 2:
                    warn(f"Something wonky about the line specification {lineinfo}") 
                dtype_prop, dtype = lineinfo['remainder']
                path.append(dtype_prop)
                object_p.make_data_path(path, parent, 1, dtype)
            elif lineinfo['type'] in 'bg':
                # It is a group that will have sub-assertions
                object_p.make_group(path, parent, 1)
            else:
                warn(f"Unable to determine object path type for {lineinfo}; no field created!")
            created_paths.append(object_p)

        return created_paths


    def make_authority_legs(self, parent, authclass):
        """Takes a parent group that is an assertion, e.g. star:E13_crm_P108 or 
        crm:E15_Identifier_Assignment. Returns either one or two STAR leg paths for the
        authority, depending on what class(es) the authority should be."""
        if authclass == 'crm:E39_Actor' and self.expand_actors:
            # This is the one that needs two paths
            return [self.make_authority_legs(parent, 'crm:E21_Person')[0], 
                    self.make_authority_legs(parent, 'spec:Author_Group')[0]]
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
            authority = self.pathbuilder.add_path(label, name)
            apath = ['crm:P14_carried_out_by', authclass]
            # The authority is always an autocomplete field
            authority.make_entityref_path(apath, parent, 1, 'reference')
            # Return a list for consistency
            return [authority]
        

    def _make_starleg(self, parent, suffix, label, predicate, object, fieldtype):
        """Internal method for source and provenance legs"""
        wpid = re.sub(r'^g_(.*)_assertion', r'p_\1', parent.id) + suffix
        wp = self.pathbuilder.add_path(wpid, label)
        wp.make_entityref_path([predicate, object], parent, 1, fieldtype)
        return wp


    def make_source_leg(self, parent):
        """Takes a parent group that is an assertion and returns its source"""
        # The source passage is always an inline form
        return self._make_starleg(parent, '_src', 'Found in', '^crm:P67_refers_to', 'crm:E33_Linguistic_Object', 'inline')


    def make_provenance_leg(self, parent):
        """Takes a parent group that is an assertion and returns its provenance"""
        # The bibliography is always an autocomplete TODO really?
        return self._make_starleg(parent, '_based', 'Based on', 'crm:P17_was_motivated_by', 'spec:Bibliography', 'reference')


    def make_assignment_paths(self, parent, lineinfo):
        """Takes a parent group that is an object, e.g. crm:E21_Person or crm:E67_Birth.
        Returns the assertion and its appropriate STAR legs based on the line info for
        object, authority, source, and provenance."""
        # Get the assertion group itself
        assertion, chainbottom = self.make_assertion_path(parent, lineinfo)
        # Then the path(s) for the object
        assignment = self.make_object_legs(assertion, lineinfo, chainbottom)
        # Prepend the assertion to the object
        assignment.insert(0, assertion)
        # If this is a grouping for further assertions, we are done after the object
        if lineinfo['type'] in 'bg':
            return assignment

        # Then the path for the authority
        if lineinfo['type'] == 'identifier':
            authorities = self.make_authority_legs(assertion, 'lrmoo:F11_Corporate_Body')
            # Add the authority and then we are done
            assignment.extend(authorities)
            return assignment
        else:
            authorities = self.make_authority_legs(assertion, 'crm:E39_Actor')
            assignment.extend(authorities)

        # Finally, the paths for source and provenance
        assignment.append(self.make_source_leg(assertion))
        assignment.append(self.make_provenance_leg(assertion))

        # Return the lot
        return assignment

    def make_event_path(self, parent, lineinfo):
        """Takes a parent group that is an object. Returns a group that has a 
        path chain through a bare assertion."""
        # Pick out the assertion and the object
        chaintop, assertion, chainbottom = self.make_assertion_pathchain(lineinfo['predicate'])
        labelseed = f"{parent.name} {lineinfo['label']}"
        event = self.pathbuilder.add_path(self.make_id(labelseed, isgroup=True), lineinfo['label'])
        epath = [chaintop, assertion, chainbottom, lineinfo['object']]
        event.make_group(epath, parent, lineinfo['type'] == 'b')
        return event
    
    def make_toplevel(self, lineinfo):
        pid = self.make_id(lineinfo['label'], isgroup=True)
        entity = self.pathbuilder.add_path(pid, lineinfo['label'])
        entity.make_group([lineinfo['object']], None, 'Unlimited')
        return entity


### Here is the line parsing logic / main routines ###

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
        if remainder[-1] in ['reference', 'inline']:
            result['entityref'] = remainder.pop()
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
    parser = ArgumentParser(
        prog='WissKI path generator',
        description='Generates a pathbuilder XML file based on a set of paths in a spreadsheet'
    )
    parser.add_argument('-f', '--pathfile', 
                        help='A file specifying the paths to build, in XLSX format')
    parser.add_argument('-o', '--ontology', 
                        help='A file specifying the ontology the pathbuilder should use, in any format that rdflib can parse')
    parser.add_argument('-n', '--no-external', action='store_true',
                        help='Specify whether to exclude paths that point to external URIs')
    parser.add_argument('-x', '--expand-subclasses', action='store_true', 
                        help='Specify whether classes such as crm:E39_Actor should be expressed in multiple paths')
    args = parser.parse_args()
    pathspec = load_workbook(filename=args.pathfile)['paths']
    g = Graph()
    g.parse(args.ontology)

    maker = STARPathMaker(Pathbuilder(g, enable=True), expand_actors=args.expand_subclasses, no_external=args.no_external)

    # Work through the input line by line, creating the necessary paths
    stack = OrderedDict()
    first_skipped = False
    for row in pathspec.values:
        if not first_skipped:
            # Assume there is a header row and skip it
            first_skipped = True
            continue
        lineinfo = parse_line(row)
        if lineinfo is None:
            # skip blank lines
            continue
        parent = find_parent(stack, lineinfo)
        if lineinfo['type'] in 'lm':
            # It is a simple datatype path. Make it and carry on;
            # it won't be a parent to anything.
            maker.make_simple_path(parent, lineinfo)
        elif lineinfo['type'] == 'top':
            # It is a top-level group. Make it and put it on the stack
            entity = maker.make_toplevel(lineinfo)
            stack[0] = entity
        elif lineinfo['type'] in 'bg':
            # It is an event group. Make it and put it on the stack
            entity = maker.make_event_path(parent, lineinfo)
            stack[lineinfo['level']] = entity
        else:
            # It is a set of assertion paths and shouldn't ever be a parent.
            maker.make_assignment_paths(parent, lineinfo)

    # Now serialise all those paths
    print(maker.pathbuilder.serialize())