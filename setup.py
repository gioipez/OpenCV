from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "IntelligentScissors.scissors.search",
        sources=["IntelligentScissors/scissors/search.pyx"],
        include_dirs=[np.get_include()],
        language='c++'
        ),
]

setup(
    ext_modules=cythonize(extensions),
)
