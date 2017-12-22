# numpy_caching

## Summary

This python package provides a method to cache the results from functions (typically numpy-based) in persistent storage. This is primarily for use with functions that take a considerable computation time, and are likely to be run multiple times from non-identical python interpreter instances. A typical use case might be iterating over different graphical representations of data, without wishing to recompute said data, or experimenting with higher level optimisations after finalising lower level computations.

## Installation

Git clone this repository to the desired local directory, and then use as a python package.

## Usage

The package provides a function wrapper, with a simple example shown below. The cache can be enabled or disabled by the first argument (`True`: enabled).
```python
from numpy_caching import np_cache

@np_cache(True)
def expensive_function(arg1, arg2, *args, **kwargs):
    return arg1 * arg2
```

A number of parameters are available to determine the behaviour of the cache, with the function signature shown below:

```python
def np_cache(enable_cache, write_cache=True, force_update=False, compress=True, hash_method='hash'):
```

- `enable_cache`: Enable caching of function with this decorator. If `False`, cache is completely bypassed.
- `write_cache`: Create a cached result if none exists and `enable_cache` is enabled
- `force_update`: Force a cache update (ignores previously cached results)
- `compress`: True if the cache should be compressed
- `hash_method`: Either 'hash', or 'readable'. Default: 'hash'. Determines the string used as cache keys (and therefore the cache file names).


The default cache directory is `./_cache/` relative to the user's working directory. This can be changed by:

```python
from numpy_caching import set_cache_dir

set_cache_dir('/tmp/someplace/')
```

## Caveats

- This tool works best on functions which take small arguments and return medium-sized results. Due to the requirement of determining a hash of a function's arguments, there is a time penalty with large arguments. Due to the usage of persistent storage, functions which return very large arrays will consume large amounts of disk space.
- There is the possibility of hash collision or similarly unexpected behaviour, resulting in incorrect results being returned. This tool should therefore **not** be used in situations where this is an unacceptable risk.
- Supported types:
  - While this tool is designed to be compatible with most numpy and standard python types as either function arguments or return values, unpickleable return types are not supported (e.g. returning a lambda function), and some unhashable argument types may not be supported.
  - This tool supports function arguments of of type `np.ndarray`, so long as `array.data.tobytes()` is consistent, and all other arguments so long as `str(arg)` is consistent. Note therefore that dictionaries (or numpy object arrays containing dictionaries) are not supported argument types - consider using keyword arguments instead.
- Pull requests are welcome! (provided they are made under the license below)

## License (MIT)

Copyright 2018 Matt Judge

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.