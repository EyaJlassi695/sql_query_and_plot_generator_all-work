from setuptools import setup, find_packages

setup(
    name="sql_query_and_plot_generator",  # Use a meaningful, lowercase name
    version="0.1.0",
    author="Wiem AMDOUNI","Mohamed Amine BESBES","Eya JLASSI","Yassine MKAOUAR"
    author_email="amdouniwiem1@gmail.com","medaminbesbes91@gmail.com","eyajlassi2306@gmail.com","yassine.mkaouar15@gmail.com",
    description="A Python package for SQL query generation and plots generation and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EyaJlassi695/LLM-for-Data-Analysis",  # Replace with your actual GitHub repo
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "streamlit",
        "google-cloud-bigquery",
        "openai",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "sql_query_and_plot_generator=sql_query_and_plot_generator.main:main",
        ],
    },
)
