def _list(func):
    def wrap(a, **kwargs):
        if isinstance(a, list): retval = func(a, **kwargs)
        elif isinstance(a, tuple): retval = func(list(a), **kwargs)
        else: retval = func([a], **kwargs)
        return retval
    return wrap
