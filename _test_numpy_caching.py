import unittest
import numpy.testing as npt
import numpy as np
import shutil
import os
import os.path

from . import *
from . import _numpy_caching


def _clear_cache():
    # remove any existing cache
    shutil.rmtree(_numpy_caching.CACHE_DIR)
    os.mkdir(_numpy_caching.CACHE_DIR)


def _cache_files():
    return os.listdir(_numpy_caching.CACHE_DIR)


def _cache_mod_times():
    return list(os.path.getmtime(os.path.join(_numpy_caching.CACHE_DIR, p)) for p in _cache_files())


def _cache_length():
    return len(_cache_files())


class TestCaching(unittest.TestCase):

    def setUp(self):
        _clear_cache()

    def test_basic(self):
        def orig(x, y):
            nonlocal f_cnt
            f_cnt += 1
            return x * y

        args = 2, 3
        expected = 6

        f_cnt = 0
        self.assertEqual(expected, orig(*args))
        self.assertEqual(f_cnt, 1)

        # don't cache when disabled
        cached = np_cache(False)(orig)
        f_cnt = 0
        self.assertEqual(expected, cached(*args))
        self.assertEqual(f_cnt, 1)
        self.assertEqual(expected, cached(*args))
        self.assertEqual(f_cnt, 2)
        self.assertEqual(0, _cache_length())

        # cache when enabled with defaults
        cached = np_cache(True)(orig)
        f_cnt = 0
        self.assertEqual(expected, cached(*args))
        self.assertEqual(f_cnt, 1)
        self.assertEqual(expected, cached(*args))
        self.assertEqual(f_cnt, 1)
        self.assertEqual(expected, cached(*args))
        self.assertEqual(f_cnt, 1)
        self.assertEqual(1, _cache_length())

    def test_cache_arg_write_cache(self):
        def f(x, y):
            nonlocal calls
            calls += 1
            return x*y
        cf = np_cache(True, write_cache=False)(f)
        args = 2, 3
        calls = 0
        expected = 6
        self.assertEqual(expected, cf(*args))
        self.assertEqual(1, calls)
        self.assertEqual(0, _cache_length())

        self.assertEqual(expected, cf(*args))
        self.assertEqual(2, calls)
        self.assertEqual(0, _cache_length())

    def test_cache_arg_force_update(self):
        def f(x, y):
            nonlocal calls
            calls += 1
            return x*y
        cf = np_cache(True, force_update=True)(f)
        args = 2, 3
        calls = 0
        expected = 6
        self.assertEqual(expected, cf(*args))
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime1 = _cache_mod_times()[0]

        self.assertEqual(expected, cf(*args))
        self.assertEqual(2, calls)
        self.assertEqual(1, _cache_length())
        mtime2 = _cache_mod_times()[0]
        self.assertNotEqual(mtime1, mtime2)

    def test_cache_arg_readable(self):
        def f(x, y):
            nonlocal calls
            calls += 1
            return x*y
        cf = np_cache(True, hash_method='readable')(f)
        args = 2, 3
        calls = 0
        expected = 6
        self.assertEqual(expected, cf(*args))
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime1 = _cache_mod_times()[0]

        self.assertEqual(expected, cf(*args))
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime2 = _cache_mod_times()[0]
        self.assertEqual(mtime1, mtime2)

    def test_numpy_args(self):
        def f(x, y):
            nonlocal calls
            calls += 1
            return x+y
        cf = np_cache(True)(f)
        args = np.arange(0, 10), np.arange(20, 30)
        calls = 0
        expected = f(*args)
        calls = 0
        self.assertTrue(np.array_equal(expected, cf(*args)))
        self.assertEqual(1, calls)
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime1 = _cache_mod_times()[0]

        self.assertTrue(np.array_equal(expected, cf(*args)))
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime2 = _cache_mod_times()[0]
        self.assertEqual(mtime1, mtime2)

    def test_numpy_large_args(self):
        def f(x, y):
            nonlocal calls
            calls += 1
            return x+y
        cf = np_cache(True)(f)
        args = np.random.rand(10, 100, 1000), np.random.rand(10, 100, 1000)
        calls = 0
        expected = f(*args)
        calls = 0
        self.assertTrue(np.array_equal(expected, cf(*args)))
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime1 = _cache_mod_times()[0]

        self.assertTrue(np.array_equal(expected, cf(*args)))
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime2 = _cache_mod_times()[0]
        self.assertEqual(mtime1, mtime2)

    def test_no_arguments(self):
        def f():
            nonlocal calls
            calls += 1
            return 5
        cf = np_cache(True)(f)
        calls = 0
        expected = 5
        self.assertEqual(expected, cf())
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime1 = _cache_mod_times()[0]

        self.assertEqual(expected, cf())
        self.assertEqual(1, calls)
        self.assertEqual(1, _cache_length())
        mtime2 = _cache_mod_times()[0]
        self.assertEqual(mtime1, mtime2)

    def test_kw_arguments(self):
        # following behaviour of functools.lru_cache
        # PEP 468: Preserving Keyword Argument Order

        def f(x, y):
            nonlocal calls
            calls += 1
            return x * y
        cf = np_cache(True)(f)
        calls = 0
        expect = f(2, 3)
        calls = 0
        self.assertEqual(expect, cf(2, 3))
        self.assertEqual(1, calls)
        self.assertEqual(expect, cf(2, y=3))
        self.assertEqual(expect, cf(x=2, y=3))
        self.assertEqual(expect, cf(y=3, x=2))
        self.assertEqual(4, calls)
        self.assertEqual(4, _cache_length())

    def test_return_values(self):
        retvals = (
            None,
            23,
            np.arange(5),
            (np.arange(5), np.arange(6)),
            (24, np.arange(4)),
            (23, (23, np.arange(5))),
            {1: 2, 3: 4},
            (23, ({1: 2, 4: 3}, np.arange(4)), np.eye(3))
        )
        for retval in retvals:
            def f():
                nonlocal calls
                calls += 1
                return retval

            cf = np_cache(True)(f)
            calls = 0

            try:
                npt.assert_equal(retval, cf())
                npt.assert_equal(retval, cf())
            except AssertionError:
                self.fail()

            self.assertEqual(1, calls)
            self.assertEqual(1, _cache_length())
            _clear_cache()


def time_large_arg_hash():
    import timeit
    setup = """\
from __main__ import _numpy_caching, np
args = np.random.rand(10, 100, 1000), np.random.rand(10, 100, 1000)
# args = np.random.rand(10, 1), np.random.rand(10, 1)
def func(a, b):
    return
    """
    num = 100

    # stmt = """_numpy_caching._func_hash_md5_fast(func, args, {})"""
    # t = timeit.timeit(stmt, setup=setup, number=num)
    # print('{} {} ms per iteration'.format(stmt, 1000*t/num))

    stmt = """_numpy_caching._func_hash_md5(func, args, {})"""
    t = timeit.timeit(stmt, setup=setup, number=num)
    print('{} {} ms per iteration'.format(stmt, 1000*t/num))


if __name__ == '__main__':
    # time_large_arg_hash()
    unittest.main()
