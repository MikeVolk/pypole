from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pypole",
    version="0.1.0a",
    author="<Your Name>",
    author_email="<Your Email>",
    description="A package to calculate magnetic field maps from magnetic dipoles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<Your Package URL>",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        # Add other required packages here
    ],
)
