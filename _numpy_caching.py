import logging

import os
import errno
from functools import wraps
from inspect import getcallargs
from hashlib import md5
from zipfile import BadZipFile
import numpy as np

CACHE_DIR = "./_cache/"


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_cache_dir(path):
    global CACHE_DIR
    CACHE_DIR = path
    make_dir(CACHE_DIR)


def _save_numpy(path, values, compress):
    arr_vals = np.array({0: values})  # maintain object structure
    if compress:
        return np.savez_compressed(path, arr_vals)
    else:
        return np.savez(path, arr_vals)


def _load_numpy(path):
    with np.load(path) as npzfile:
        # get the array of the first (only) stored 'file'
        cached_item = npzfile.items()[0][1]
        # get the value of key used in _save_numpy
        res = cached_item.item()[0]
        return res


def _trim_str_len(s, max_len):
    return s[:max_len] if len(s) > max_len else s


def _func_hash_md5(func, args, kwargs):
    callargs = getcallargs(func, *args, **kwargs)
    hashargs = tuple(
        (k,
         md5(v.data.tobytes()).hexdigest() if isinstance(v, np.ndarray) else md5(str(v).encode()).hexdigest()
         ) for k, v in sorted(callargs.items())
    )
    return md5(str((func.__name__, hashargs)).encode()).hexdigest()


def _func_hash_readable(func, args, kwargs):
    max_var_len = 20
    max_filename_len = 100

    def stringify_var(var):
        # remove non-alphanumeric characters
        s = ''.join(x for x in str(var) if x.isalnum())
        return _trim_str_len(s, max_var_len)

    slug = '{fnm}_{args}_{kwargs}'.format(
        fnm=func.__name__,
        args='_'.join(map(stringify_var, args)),
        kwargs='_'.join('_'.join(map(stringify_var, kw)) for kw in kwargs.items())
    )
    return '{slug}_{hash}'.format(
        slug=_trim_str_len(slug, max_filename_len - 11),  # minus hash and file extension
        hash=_func_hash_md5(func, args, kwargs)[:6]
    )


def np_cache(enable_cache, write_cache=True, force_update=False, compress=True, hash_method='hash'):
    """
    Cache any function that has hashable (or string representable) arguments and returns a numpy object
    :param enable_cache: Enable caching of function with this decorator
    :param write_cache: Create a cached result if none exists and enable_cache is enabled
    :param force_update: Force a cache update (ignores previously cached results)
    :param compress: True if the cache should be compressed
    :param hash_method: Either 'hash', or 'readable'. Default: 'hash'
    :return: The function result, cached if use_cache is enabled
    """

    valid_hash_funcs = {
        'hash': _func_hash_md5,
        'readable': _func_hash_readable
    }
    try:
        hash_func = valid_hash_funcs[hash_method]
    except KeyError:
        msg = "hash_method argument value must be one of {}".format(', '.join(valid_hash_funcs.keys()))
        raise ValueError(msg) from None

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enable_cache:
                # Don't cache anything
                return func(*args, **kwargs)
            # create cache file path
            hash_key = '{}.npz'.format(hash_func(func, args, kwargs))
            cache_path = os.path.join(CACHE_DIR, hash_key)

            def run_func_update_cache():
                res = func(*args, **kwargs)
                if force_update or write_cache:
                    # logging.debug("Writing to cache")
                    _save_numpy(cache_path, res, compress)
                return res

            if force_update:
                logging.debug("Cache: Forcing update on {}".format(hash_key))
                return run_func_update_cache()
            else:
                try:
                    result = _load_numpy(cache_path)
                    logging.debug("Cache: Found {}".format(hash_key))
                    return result
                except (IOError, FileNotFoundError):
                    logging.debug("Cache: Not found {}".format(hash_key))
                    return run_func_update_cache()
                except BadZipFile:
                    logging.warning("Cache: Corrupted file, ignoring {}".format(hash_key))
                    return run_func_update_cache()
        return wrapper

    return decorator


# ensure cache directory exists
make_dir(CACHE_DIR)


if __name__ == '__main__':
    pass
