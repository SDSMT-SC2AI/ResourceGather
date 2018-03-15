# robert king @ stackoverflow.com
def dump_args(func):
    "This decorator dumps out the arguments passed to a function before calling it"
    length = func.__code__.co_argcount
    argnames = func.__code__.co_varnames[:length]
    fname = func.__name__

    def echo_func(*args, **kwargs):
        print(fname, "(", '\n'.join(
            '{}={}'.format(k, v) for k, v in
            list(zip(argnames, args[:len(argnames)]))
            + [("args", list(args[len(argnames):]))]
            + [("kwargs", kwargs)]) + ")")
    return echo_func