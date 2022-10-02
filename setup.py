import setuptools
from pathlib import Path

setuptools.setup(
    name="pycan",
    version=1.0,
    author="Armin Abbasi Najarzadeh",
    author_email="apmnh.abbasi@gmail.com",
    description="A package associated with extracting financial data from yahoo finance and performing financial analysis.",
    long_description=Path("README.md").read_text(),
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'yfinance',
                      'matplotlib', 'numpy', 'scipy', 'seaborn']
)
