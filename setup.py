from setuptools import setup, find_packages

setup(
    name="idkdoe",
    version="0.1.0",
    description="Librería de diseño de experimentos.",
    author="Ander Alvarez Sanz",
    author_email='andersz.alvarez@gmail.com',  # Replace with your email
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
