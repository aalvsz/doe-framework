from setuptools import setup, find_packages

setup(
    name="idkDOE",
    version="0.1.0",
    description="Librería de diseño de experimentos.",
    author="Ander Alvarez Sanz",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
