class CLRParser:
    productions = []
    grammar = []
    EPSILON = "e"
    terminals = set()
    non_terminals = set()
    vars_in_rhs = set()
    stateDict = dict()

    def __init__(self, grammar):
        self.grammar = grammar
        self.get_symbols()
        self.augmentFirstState()
        self.appendDotToGrammar()
        self.arrayBindingVariablesWithRules()

  
    def get_symbols(self):
        for production in self.grammar:
            lhs, rhs = production.split('->')
            # Set of non terminals only
            self.non_terminals.add(lhs.strip())
            # iterate through rhs chars: treat adjacent 'i'+'d' as single token 'id'
            i = 0
            while i < len(rhs):
                ch = rhs[i]
                # skip spaces
                if ch == ' ':
                    i += 1
                    continue
                # handle "id" as single terminal
                if ch == 'i' and i + 1 < len(rhs) and rhs[i+1] == 'd':
                    self.terminals.add('id')
                    i += 2
                    continue
                # add single character terminal if it's not a non-terminal
                if ch not in self.non_terminals and not ch.isupper() and ch != '|':
                    # print("adding ", self.non_terminals, ch)
                    self.terminals.add(ch)
                i += 1

                if(ch.isupper()):
                    self.vars_in_rhs.add(ch)
        # Remove the non terminals
        # terminals = self.terminals.difference(self.non_terminals)
        self.terminals = list(self.terminals)
        self.terminals.append('$')
        

        return self.terminals, self.non_terminals, self.vars_in_rhs
    
    # validate if vars_in_rhs is subset of lhs
    def validate_grammar(self):
        a, non_terminals, vars_in_rhs = self.get_symbols()
        if vars_in_rhs.issubset(non_terminals):
            return True
        else:
            return False

    def augmentFirstState(self):
        # This function will add a new starting production
        start_symbol = self.grammar[0].split('->')[0].strip()
        new_production = start_symbol+"' -> " + start_symbol
        self.grammar.insert(0, new_production)
        return self.grammar    

    def appendDotToGrammar(self):
        # This function will append dot to each production
        new_grammar = []

        for production in self.grammar:
            lhs, rhs = production.split('->')
            rhs = rhs.split('|')
            new_production = []
            for item in rhs:
                item = '.' + item.strip()
                # new_production.append(item)
                new_production = '->'.join([lhs.strip(), item])
                new_grammar.append(new_production)
                # print(new_grammar)
            # new_production = ' | '.join(new_production)
            # new_grammar.append('->'.join([lhs.strip(), new_production]))
        self.grammar = new_grammar
        # print(self.grammar)
        return self.grammar

    def arrayBindingVariablesWithRules(self):
        # This function will bind the variables with their productions
        prod_map = {}
        for production in self.grammar:
            lhs, rhs = production.split('->')
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs not in prod_map:
                prod_map[lhs] = []
            prod_map[lhs].append(rhs)

        # Join multiple RHS alternatives with ' | ' for each LHS
        self.productions = [[lhs, ' | '.join(rhs_list)] for lhs, rhs_list in prod_map.items()]
        return self.productions

    # --- LR(1) implementation  ---
    class Item:
        def __init__(self, lhs, rhs, dot, look):
            self.lhs = lhs
            self.rhs = tuple(rhs)
            self.dot = dot
            self.look = look

        def next_symbol(self):
            if self.dot < len(self.rhs):
                return self.rhs[self.dot]
            return None

        def advance(self):
            return CLRParser.Item(self.lhs, self.rhs, self.dot + 1, self.look)

        def __eq__(self, other):
            return isinstance(other, CLRParser.Item) and (self.lhs, self.rhs, self.dot, self.look) == (other.lhs, other.rhs, other.dot, other.look)

        def __hash__(self):
            return hash((self.lhs, self.rhs, self.dot, self.look))

        def __repr__(self):
            before = ''.join(self.rhs[:self.dot])
            after = ''.join(self.rhs[self.dot:])
            return f"{self.lhs} -> {before}.{after} , {self.look}"

    def _tokenize_rhs(self, rhs_str):
        #  Tokenize a compact RHS (no spaces), treating 'id' specially and 'e' as epsilon (empty RHS).
        s = rhs_str.strip()
        if s == self.EPSILON:
            return []
        tokens = []
        i = 0
        while i < len(s):
            if s[i] == 'i' and i + 1 < len(s) and s[i+1] == 'd':
                tokens.append('id')
                i += 2
            else:
                tokens.append(s[i])
                i += 1
        return tokens

    def _all_productions(self):
        #  Return list of (lhs, rhs_tuple) for each alternative (removes the leading dot if present).
        prods = []
        for lhs, rhs_combined in self.productions:
            parts = [p.strip() for p in rhs_combined.split('|')]
            for p in parts:
                # remove leading dot if present
                p = p.lstrip('.')
                tokens = self._tokenize_rhs(p)
                prods.append((lhs, tuple(tokens)))
        return prods

    def compute_first(self):
        """Compute FIRST sets for symbols based on current productions."""
        prods = self._all_productions()
        # gather non-terminals and terminals
        nonterms = [lhs for lhs, _ in prods]
        terms = set()
        for _, rhs in prods:
            for sym in rhs:
                if sym.isupper():
                    if sym not in nonterms:
                        nonterms.append(sym)
                else:
                    terms.add(sym)
        FIRST = {nt: set() for nt in nonterms}
        for t in terms:
            FIRST[t] = {t}
        FIRST[self.EPSILON] = {self.EPSILON}

        changed = True
        while changed:
            changed = False
            for A, rhs in prods:
                if len(rhs) == 0:
                    if self.EPSILON not in FIRST[A]:
                        FIRST[A].add(self.EPSILON)
                        changed = True
                    continue
                add_eps = True
                for X in rhs:
                    X_first = FIRST.get(X, set())
                    before = len(FIRST[A])
                    FIRST[A].update(x for x in X_first if x != self.EPSILON)
                    if self.EPSILON not in X_first:
                        add_eps = False
                        break
                    if len(FIRST[A]) != before:
                        changed = True
                if add_eps:
                    if self.EPSILON not in FIRST[A]:
                        FIRST[A].add(self.EPSILON)
                        changed = True
        self.FIRST = FIRST
        return FIRST

    def _first_of_sequence(self, seq):
        # Compute FIRST set for a sequence (list/tuple) of symbols.
        res = set()
        for X in seq:
            X_first = self.FIRST.get(X, {X})
            res.update(x for x in X_first if x != self.EPSILON)
            if self.EPSILON not in X_first:
                return res
        res.add(self.EPSILON)
        return res

    def closure_lr1(self, items):
        # Compute LR(1) closure for a set of items.
        prods = self._all_productions()
        prod_map = {}
        for A, rhs in prods:
            prod_map.setdefault(A, []).append(rhs)

        closure = set(items)
        changed = True
        while changed:
            changed = False
            new_items = set(closure)
            for it in list(closure):
                if it.dot >= len(it.rhs):
                    continue
                B = it.rhs[it.dot]
                if not B.isupper():
                    continue
                beta = it.rhs[it.dot+1:]
                for gamma in prod_map.get(B, []):
                    seq = list(beta) + [it.look]
                    first_set = self._first_of_sequence(seq)
                    for b in first_set:
                        item = CLRParser.Item(B, gamma, 0, b)
                        if item not in new_items:
                            new_items.add(item)
                            changed = True
            closure = new_items
        return frozenset(closure)

    def goto_lr1(self, items, X):
        moved = set()
        for it in items:
            if it.dot < len(it.rhs) and it.rhs[it.dot] == X:
                moved.add(CLRParser.Item(it.lhs, it.rhs, it.dot+1, it.look))
        if not moved:
            return frozenset()
        return self.closure_lr1(moved)

    def build_canonical_collection(self):
        # Build LR(1) canonical collection from self.productions format.
        

        # compute FIRST
        self.compute_first()
        prods = self._all_productions()
        if not prods:
            self.states = []
            self.transitions = {}
            return self.states, self.transitions
        # augmented start is assumed to be the first production's LHS
        start_lhs = self.productions[0][0]
        # pick first rhs for start
        start_rhs = prods[0][1]
        start_item = CLRParser.Item(start_lhs, start_rhs, 0, '$')
        I0 = self.closure_lr1({start_item})
        states = [I0]
        transitions = {}
        queue = [I0]
        while queue:
            I = queue.pop(0)
            symbols = set()
            for it in I:
                if it.dot < len(it.rhs):
                    symbols.add(it.rhs[it.dot])
            for X in symbols:
                J = self.goto_lr1(I, X)
                if not J:
                    continue
                if J not in states:
                    states.append(J)
                    queue.append(J)
                i = states.index(I)
                j = states.index(J)
                transitions[(i, X)] = j
        self.states = states
        self.transitions = transitions
        return states, transitions

    def pretty_print_states(self):
        for idx, I in enumerate(self.states):
            print(f"State I{idx}:")
            for it in sorted(I, key=lambda x: (x.lhs, x.rhs, x.dot, x.look)):
                print('  ', it)
            print()

    def find_production_index(self, lhs, rhs):
        # Return the index of production (lhs, rhs) within the flattened production list.
        prods = self._all_productions()
        for i, (A, B) in enumerate(prods):
            if A == lhs and B == rhs:
                return i
        return -1

    def build_parsing_table(self):
        # Build ACTION and GOTO tables (dicts) and optional pandas DataFrames.
        
        prods = self._all_productions()
        nonterms = set(A for A, _ in prods)
        terms = set()
        for _, rhs in prods:
            for s in rhs:
                if not s.isupper():
                    terms.add(s)
        terms.add('$')

        from collections import defaultdict
        self.action = defaultdict(dict)
        self.goto = defaultdict(dict)

        for i, I in enumerate(self.states):
            for it in I:
                a = it.next_symbol()
                if a is None:
                    # reduction or accept
                    if it.lhs == self.productions[0][0]:
                        self.action[i]['$'] = ('acc',)
                    else:
                        prod_no = self.find_production_index(it.lhs, it.rhs)
                        self.action[i][it.look] = ('r', prod_no, it.lhs, it.rhs)
                else:
                    if a in terms:
                        j = self.transitions.get((i, a))
                        if j is not None:
                            self.action[i][a] = ('s', j)
                    else:
                        j = self.transitions.get((i, a))
                        if j is not None:
                            self.goto[i][a] = j

        # Try to build pandas DataFrames for nice display (optional)
        try:
            import pandas as pd
            terminals_sorted = sorted(list(terms))
            ACTION = pd.DataFrame('', index=range(len(self.states)), columns=terminals_sorted)
            GOTO = pd.DataFrame('', index=range(len(self.states)), columns=sorted(list(nonterms)))
            for i in range(len(self.states)):
                for a, act in self.action[i].items():
                    if act[0] == 's':
                        ACTION.at[i, a] = f'S{act[1]}'
                    elif act[0] == 'r':
                        ACTION.at[i, a] = f'R{act[1]}'
                    elif act[0] == 'acc':
                        ACTION.at[i, a] = 'acc'
                for A, j in self.goto.get(i, {}).items():
                    GOTO.at[i, A] = j
            self.ACTION_DF = ACTION
            self.GOTO_DF = GOTO
        except Exception:
            self.ACTION_DF = None
            self.GOTO_DF = None

        return self.action, self.goto

    def print_parsing_table(self):
        """Print parsing table; prefers pandas DataFrames if available."""
        if getattr(self, 'ACTION_DF', None) is not None and getattr(self, 'GOTO_DF', None) is not None:
            print('ACTION table:')
            print(self.ACTION_DF)
            print('\nGOTO table:')
            print(self.GOTO_DF)
            return
        print('ACTION (dict):')
        for i in range(len(self.states)):
            row = self.action.get(i, {})
            formatted = {k: (f'S{v[1]}' if v[0]=='s' else (f'R{v[1]}' if v[0]=='r' else 'acc')) for k,v in row.items()}
            print(f'I{i}:', formatted)
        print('\nGOTO (dict):')
        for i in range(len(self.states)):
            row = self.goto.get(i, {})
            print(f'I{i}:', row)

    def draw_parsetree(self, nodes, edges, filename='parsetree'):
        # Draw parse tree as PNG using graphviz or the system 'dot' executable.
        
        # Helper to safely escape labels
        def _esc(s):
            return str(s).replace('"', '\\"')

        
        try:
            from graphviz import Digraph
            from graphviz.backend.execute import ExecutableNotFound
            dot = Digraph('parsetree', format='png')
            for nid, lbl in nodes.items():
                dot.node(str(nid), _esc(lbl))
            for a, b in edges:
                dot.edge(str(a), str(b))
            try:
                out = dot.render(filename, cleanup=True)
                return out
            except ExecutableNotFound:
                # graphviz python binding found but dot executable missing; fall through to system dot attempt
                dot_source = dot.source
        except Exception:
            # graphviz package not available; we'll write DOT source ourselves
            dot_source = ['digraph parse {']
            for nid, lbl in nodes.items():
                dot_source.append(f'  n{nid} [label="{_esc(lbl)}"];')
            for a, b in edges:
                dot_source.append(f'  n{a} -> n{b};')
            dot_source.append('}')
            dot_source = '\n'.join(dot_source)

        # At this point we have DOT source in dot_source (string). Write it to a .gv file.
        gvpath = filename if filename.endswith('.gv') else filename + '.gv'
        with open(gvpath, 'w', encoding='utf-8') as f:
            f.write(dot_source)

        # Rendering to PNG via system 'dot' is intentionally omitted.
        # Return the DOT file path; if the python 'graphviz' binding succeeded earlier it would have
        # returned a rendered file; otherwise we provide the .gv file for user-controlled rendering.
        import os
        return os.path.abspath(gvpath)
    
    
    def _tokenize_input(self, inp):
        # Tokenize input string or accept list of tokens.
        # If inp is a list, return it. If string contains spaces, split. Otherwise, tokenize like _tokenize_rhs.
        
        if isinstance(inp, list):
            return list(inp)
        if isinstance(inp, str):
            s = inp.strip()
            if ' ' in s:
                return [tok for tok in s.split() if tok]
            # compact string: tokenize characters / id
            return self._tokenize_rhs(s)
        raise TypeError('Input must be list or string')

    def parse_string(self, inp, display=True, draw=False, filename='parsetree'):
        # Parse input; return (accepted: bool, steps: list, nodes: dict, edges: list).
        if not hasattr(self, 'states') or not self.states:
            self.build_canonical_collection()
        if not hasattr(self, 'action') or not self.action:
            self.build_parsing_table()

        tokens = self._tokenize_input(inp)
        # ensure $ at end
        if not tokens or tokens[-1] != '$':
            tokens = tokens + ['$']

        stack = [0]
        ip = 0
        steps = []

        # parse tree data
        nodes = {}   # node_id -> label
        edges = []   # (parent_id, child_id)
        symbol_stack = []  # stack of node_ids corresponding to grammar symbols
        node_id = 0

        while True:
            s = stack[-1]
            a = tokens[ip]
            action = self.action.get(s, {}).get(a)
            stack_str = ' '.join(map(str, stack))
            input_str = ' '.join(tokens[ip:])
            action_str = ''
            if action is None:
                action_str = 'error'
                steps.append((stack_str, input_str, action_str))
                if display:
                    for row in steps:
                        print(row)
                    print('\nString rejected: no action for', (s, a))
                return False, steps, nodes, edges
            if action[0] == 's':
                action_str = f'shift S{action[1]}'
                # create leaf node for terminal
                nodes[node_id] = a
                symbol_stack.append(node_id)
                node_id += 1

                stack.append(a)
                stack.append(action[1])
                ip += 1
            elif action[0] == 'r':
                _, pidx, A, rhs = action
                action_str = f'reduce by {A} -> {"".join(rhs) if rhs else "ε"}'
                children = []
                if len(rhs) > 0:
                    for _ in range(len(rhs)):
                        stack.pop()  # pop state
                        nid = symbol_stack.pop()
                        stack.pop()  # pop symbol
                        children.insert(0, nid)
                else:
                    # epsilon production: create an epsilon node
                    nodes[node_id] = 'ε'
                    children = [node_id]
                    node_id += 1

                t = stack[-1]
                goto_state = self.goto.get(t, {}).get(A)
                if goto_state is None:
                    steps.append((stack_str, input_str, 'error: no goto'))
                    if display:
                        for row in steps:
                            print(row)
                        print('\nString rejected: no GOTO for', (t, A))
                    return False, steps, nodes, edges

                # create a node for the LHS and connect to children
                nodes[node_id] = A
                for c in children:
                    edges.append((node_id, c))
                symbol_stack.append(node_id)
                node_id += 1

                stack.append(A)
                stack.append(goto_state)
            else:  # accept
                action_str = 'accept'
                steps.append((stack_str, input_str, action_str))
                if display:
                    for row in steps:
                        print(row)
                    print('\nString accepted')
                # draw tree if requested
                out_path = None
                if draw and nodes:
                    out_path = self.draw_parsetree(nodes, edges, filename=filename)
                    if display:
                        print('\nParse tree output:', out_path)
                    return True, steps, nodes, edges, out_path
                return True, steps, nodes, edges
            steps.append((stack_str, input_str, action_str))

        # unreachable
        return False, steps, nodes, edges


    
