from setuptools import setup, find_packages
import os

# Function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='exact_simplex',
    version='0.1.1', # Incremented version slightly
    author='Jules Agent', # Placeholder
    author_email='josh.t.burdick@gmail.com',
    description='An exact simplex algorithm implementation using Python Fractions.',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='http://example.com/p/exact_simplex', # Placeholder URL
    packages=find_packages(exclude=['tests*']), # Exclude tests directory
    install_requires=[
        # No external dependencies, as it uses standard library 'fractions'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Assuming MIT License from initial repo
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='linear programming, simplex algorithm, optimization, operations research, exact arithmetic, fractions',
    python_requires='>=3.6',
    project_urls={ # Optional
        'Source': 'http://example.com/p/exact_simplex_solver/source/', # Placeholder
        'Tracker': 'http://example.com/p/exact_simplex_solver/issues/', # Placeholder
    },
)
