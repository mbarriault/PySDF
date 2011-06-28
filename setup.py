from distutils.core import setup, Extension

module1 = Extension('pysdf', include_dirs=['/usr/local/include'], libraries = ['bbhutil'], library_dirs=['/usr/local/lib'], sources = ['pysdfmodule.cpp'])

setup (name = 'PySDF',
       version = '1.0',
       description = 'Python wrapper for some bbhutil routines',
       ext_modules = [module1])