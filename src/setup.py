from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "filters_cython",
        ["filters_cython.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="Cythonized Filters",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
