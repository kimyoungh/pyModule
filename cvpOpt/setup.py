from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
# Something weird....

setup(
	ext_modules = cythonize("cvpOpt.pyx"),
	include_dirs = [np.get_include()]
)
