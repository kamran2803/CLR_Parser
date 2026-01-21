# CLR Parser Implementation
# Author: Kamran Ahmed
# Date: 2026

from grammar import Grammar
import sys
from clr import CLRParser 


grammar = Grammar()

grammar =grammar.grammar_input()

gtext = '\n'.join(grammar)

# grammar = ['S -> AA' , 'A -> aA | b']
# grammar = ['S -> AaAb | BbAa', 'A -> e', 'B -> e']
# grammar = ['S -> Aa | bAc | Bc| bBa','A ->d','B -> d']

p = CLRParser(grammar)



if __name__ == '__main__':
    
    
    print('\nProductions from arrayBindingVariablesWithRules:')
    print(p.productions)

    print('\nFIRST sets:')
    p.compute_first()
    for k in sorted(p.FIRST.keys()):
        print(f"{k:8} -> {p.FIRST[k]}")
    
    print('\nCanonical collection (states):')
    p.build_canonical_collection()
    p.pretty_print_states()
    
    print('Transitions:')
    for (i, X), j in p.transitions.items():
        print(f'I{i} -- {X} --> I{j}')
    

    print('\nParsing Table (ACTION / GOTO):')
    p.build_parsing_table()
    p.print_parsing_table()

    

    print('\nEnter input string tokens separated by spaces (terminals) or as a compact string (e.g., "aab") :')
    s = input().strip()
    result = p.parse_string(s, display=True, draw=True, filename='parsetree')
    # parse_string now returns either (accepted, steps, nodes, edges) or (accepted, steps, nodes, edges, out_path)
    if len(result) == 5:
        accepted, steps, nodes, edges, out_path = result
    else:
        accepted, steps, nodes, edges = result
        out_path = None
    print('\nResult:', 'Accepted' if accepted else 'Rejected')
    if accepted and out_path:
        print('Parse tree written to:', out_path)

    try:
        import os, shutil, subprocess
        if out_path and out_path.endswith('.gv'):
            pngpath = 'parsetree.png'
            dot_exec = shutil.which('dot')
            print("dot_exec:", dot_exec)
            if dot_exec:
                subprocess.run([dot_exec, '-Tpng', out_path, '-o', pngpath], check=True)
                print("Parse tree PNG generated at:", os.path.abspath(pngpath))
    except Exception as identifier:
        pass