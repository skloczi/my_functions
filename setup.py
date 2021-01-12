import setuptools


setuptools.setup(
    name="myfunctions", # Replace with your own username
    version="0.0.1",
    author="Anna Maria Sklodowska",
    author_email="anna.maria.sklodowska@gmail.com",
    description="A small example package",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)