from setuptools import setup, find_packages

setup(
    name="QRT",
    version="0.1",
    description="QRT Trading Competition",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "cvxpy",
        "python-dotenv",
        "scikit-learn",
        "yfinance",
        "pyarrow",
    ]
)