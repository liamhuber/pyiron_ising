"""
Setuptools based setup module
"""
from setuptools import setup, find_packages

setup(
    name='pyiron_ising',
    version='1.0',
    description='Ising model code based on the work in the MSc thesis of Vijay Bhuva.',
    long_description='http://pyiron.org',

    url='https://github.com/liamhuber/pyiron_ising',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='huber@mpie.de',
    license='BSD',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9'
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        'matplotlib==3.4.3',
        'pyiron_atomistics==0.2.27'
    ],
    extras_require=[
        'nglview==2.7.7',
        'seaborn'
    ],

)
