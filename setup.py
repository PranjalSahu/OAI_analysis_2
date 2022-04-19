from setuptools import setup, find_packages
from oai_analysis_2 import __version__

setup(
    name='oai_package',
    version=__version__,
    url='https://github.com/PranjalSahu/OAI_analysis_2.git',
    author='Pranjal Sahu',
    author_email='pranjalsahu5@gmail.com',
    packages=find_packages(exclude=["test", "*.tests", "*.tests.*", "tests.*"]),
)
