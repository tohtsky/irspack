import os
from pathlib import Path
from typing import Any

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "Readme.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "numpy>=1.12.0",
        "fastprogress>=0.2",
        "optuna>=2.5.0",
        "pandas>=1.0.0",
        "scipy>=1.0",
        "colorlog>=4",
        "pydantic>=1.8.2",
        "typing_extensions>=3.10",
    ],
)


def local_scheme(version: Any) -> str:
    return ""


setup(
    name="irspack",
    # version=get_version(),
    url="https://irspack.readthedocs.io/",
    use_scm_version={
        "local_scheme": local_scheme
    },  # https://github.com/pypa/setuptools_scm/issues/342
    author="Tomoki Ohtsuki",
    author_email="tomoki.otsuki129@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages("src"),
    python_requires=">=3.7",
    package_dir={"": "src"},
)
