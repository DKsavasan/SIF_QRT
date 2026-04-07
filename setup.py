from setuptools import setup, find_packages

setup(
    name="QRT",
    version="0.1",
    description="QRT Trading Competition",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "sqlalchemy",
        "psycopg2-binary",
        "asyncpg",
        "python-dotenv",
        "fredapi",
        "yfinance",
        "quantstats",
        "wrds",
    ],
    extras_require={
        "plots": ["matplotlib", "plotly"],
        "ml": ["scikit-learn", "sympy"],
        "notebooks": ["nbconvert", "rpy2", "streamlit"],
        "all": [
            "matplotlib",
            "plotly",
            "scikit-learn",
            "sympy",
            "nbconvert",
            "rpy2",
            "streamlit",
        ],
    },
)