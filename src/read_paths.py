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
from collections import OrderedDict
from warnings import warn
from openpyxl import load_workbook
from rdflib import Graph
from string import punctuation
from wisski_pathbuilder.Pathbuilder import Pathbuilder


class STARPathMaker:
    def __init__(self, pathbuilder, expand_actors=False, no_external=False):
        self.pathbuilder = pathbuilder
        self.expand_actors = expand_actors
        self.no_external = no_external
        # LATER make these configurable?
        self.star_subject = 'crm:P140_assigned_attribute_to'
        self.star_object = 'crm:P141_assigned'
        self.star_authority = 'crm:P14_carried_out_by'
        self.star_source = '^crm:P67_refers_to'
        self.star_based = 'crm:P17_was_motivated_by'

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
        """Takes a 'vanilla' predicate with possible ^ prefix and returns the correct STARified path chain"""
        reversed = predicate.startswith('^')
        prefix, name = predicate.lstrip('^').split(':')
        pid = name.split('_')[0]
        # s, o = ('P141', 'P140') if reversed else ('P140', 'P141') 
        # return f'star:{s}_{prefix}_{pid}', f'star:E13_{prefix}_{pid}', f'star:{o}_{prefix}_{pid}'
        if reversed:
            return '^' + self.star_object, f'star:E13_{prefix}_{pid}', self.star_subject
        else:
            return '^' + self.star_subject, f'star:E13_{prefix}_{pid}', self.star_object


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

        # Are we expanding actor classes?
        created_paths = []
        pathobjs = [lineinfo['object']]
        if self.expand_actors:
            if lineinfo['object'] == 'crm:E39_Actor':
                pathobjs = ['crm:E21_Person', 'crm:E74_Group']
            elif lineinfo['object'] == 'pwro:WE3_Embodied_Actor':
                pathobjs = ['crm:E21_Person', 'pwro:WE4_Manifest_Group']
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
        if args.no_external and lineinfo['predicate'] == 'owl:sameAs':
            # We don't make paths with external references; this includes sameAs. It breaks WissKI.
            continue
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