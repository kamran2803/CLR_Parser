import re

class Grammar:
    __allowGrammarPattern = re.compile(r'^[A-Z][\s]*->[\s]*([a-zA-Z\+\-\*\/\%\#\@\!\^|\s]*|ε)([A-Z]*)$') # Allow spaces as well if needed
    __grammer = []

    def __validate_grammar_pattern(self, grammar_lines):
        for line in grammar_lines:
            if not self.__allowGrammarPattern.match(line):
                return False
        return True
    
    def grammar_input(self):
        
        print('Enter grammar productions (one per line). Press Enter on an empty line to finish:')

        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                break
            check_grammar = self.__validate_grammar_pattern([line])
            if not check_grammar:
                print('Invalid grammar production format. Please use the format: S -> E or A -> aB etc.')
            else:
                # move multiple spaces from the string line to single space
                line = re.sub(r'\s+', ' ', line)
                self.__grammer.append(line)

        return self.__grammer