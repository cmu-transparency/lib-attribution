
from termcolor import colored

import influence_tests
import invariant_tests

def all_tests():
	print(colored('*** Influence ***', 'blue'))
	influence_tests.all_tests()
	print(colored('*** Invariants ***', 'blue'))
	invariant_tests.all_tests()