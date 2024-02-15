from setuptools import setup

setup(
    name="symexp",
    version="0.1.0",
    description="Symbolic Expression",
    url="https://github.com/milmillin/symexp",
    author="Milin Kodnongbua",
    author_email="milink@cs.washington.edu",
    license="MIT",
    packages=["symexp"],
    install_requires=["typeguard", "pydantic", "gurobipy", "scipy"],
)