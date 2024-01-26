from setuptools import setup, find_packages

__version__ = "1.0.0"

# Load README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='pgn',
    version='1.0.0',
    description='A description of your package',
    author='Zachary Gale-Day',
    author_email='z.gale.day@gmail.com',
    license="MIT",
    packages=find_packages(),
    install_requires=[],  # List your package dependencies here
)
