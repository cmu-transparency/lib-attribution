import influence_tests
import invariant_tests

def all_tests():
	print('*** Influence ***')
	influence_tests.all_tests()
	print('*** Invariants ***')
	invariant_tests.all_tests()