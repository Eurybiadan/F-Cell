from setuptools import setup, find_packages

setup(
    name='pyORG',
    version='0.1',
    package_dir={"":"ocvl"},
    packages=find_packages(where="ocvl"),
    url='',
    license='Apache 2.0',
    author='Robert F Cooper',
    author_email='robert.cooper@marquette.edu',
    description='A python library for doing ORG analyses.',
    install_requires=['matplotlib',
                      'scipy',
                      'numpy',
                      'scikit-image',
                      'pandas',
                      'SimpleITK',
                      'setuptools'],
)
