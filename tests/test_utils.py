
def run_test(t_fn, *args, **kwargs):
	r = t_fn(*args, **kwargs)
	if r:
		print('passed', t_fn.__name__)
	else:
		print('failed', t_fn.__name__)