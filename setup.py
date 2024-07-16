from setuptools import find_packages, setup

setup(
    name="InteractionMesh",
    version="0.1",
    description="Inspection of Interaction Metric based on Interaction Mesh",
    author="Geonho Leem",
    author_email="geonholeem@imo.snu.ac.kr",
    install_requires=[
        "torch==2.3.1",
        "gitpython",
        "numpy==1.23.1",
        "matplotlib",
        "aitviewer",
        "argparse",
    ],
    packages=find_packages(include=["im*"], exclude=["res", "scripts", ]),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
