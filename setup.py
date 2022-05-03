import os
from pathlib import Path
from typing import Any, List, Tuple

from pybind11.setup_helpers import Pybind11Extension, build_ext
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


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        target_dir = SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return target_dir.name

        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir.name


module_name_and_sources: List[Tuple[str, List[str]]] = [
    ("irspack.evaluator._core", ["cpp_source/evaluator.cpp"]),
    ("irspack.recommenders._ials", ["cpp_source/als/wrapper.cpp"]),
    ("irspack.recommenders._knn", ["cpp_source/knn/wrapper.cpp"]),
    ("irspack.utils._util_cpp", ["cpp_source/util.cpp"]),
]
ext_modules = [
    Pybind11Extension(module_name, sources, include_dirs=[get_eigen_include()])
    for module_name, sources in module_name_and_sources
]


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
    description="Implicit feedback-based recommender system pack",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    install_requires=install_requires,
    include_package_data=True,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    python_requires=">=3.6",
)
