from setuptools import setup, find_packages

setup(
    name="clickhouse_mcp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.25.0",
        "mcp>=1.3.0",
        "langchain-aws>=0.2.15",
    ],
    python_requires=">=3.7",
)
