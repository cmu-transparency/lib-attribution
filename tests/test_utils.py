
import colorama
from termcolor import colored
colorama.init()

def run_test(t_fn, *args, **kwargs):
	r = t_fn(*args, **kwargs)
	if r:
		print(colored('passed', color='green'), t_fn.__name__)
	else:
		print(colored('FAILED', color='red'), t_fn.__name__)