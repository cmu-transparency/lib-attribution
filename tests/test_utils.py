
from termcolor import colored

def run_test(t_fn, *args, **kwargs):
	r = t_fn(*args, **kwargs)
	if r:
		print(colored('passed', 'green'), t_fn.__name__)
	else:
		print(colored('FAILED', 'red'), t_fn.__name__)