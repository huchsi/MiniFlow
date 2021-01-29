import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MiniFlow-pkg-shc",  # Replace with your own username
    version="0.0.1",
    author="sihuachun",
    author_email="sihuachun2132@gmail.com",
    description="This is a mini but function completed neural network framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sihuachun/MiniFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)