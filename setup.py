from setuptools import setup, find_packages

setup(
    name="InteGraphAD",
    version="0.1.0",
    description="A CSDL-based numerical integration package",
    author="Sankalp Kaushik",
    author_email="kaushiksankalp02@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "csdl-alpha",
    ],
    include_package_data=True,
    license="MIT",
)