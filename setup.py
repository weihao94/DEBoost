import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deboost",
    version="0.10.0",
    author="Wei Hao Khoong",
    author_email="khoongweihao@u.nus.edu",
    description="DEBoost: A Python Library for Weighted Distance Ensembling in Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weihao94/DEBoost",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
