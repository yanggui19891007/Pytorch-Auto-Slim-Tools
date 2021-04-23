import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-pruning",
    version="1.0",
    author="liangyingping",
    author_email="1691608003@qq.com",
    description="A pytorch toolkit for pruning automatically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch'],
    python_requires='>=3.6',
)