import os
import sys
from typing import Any, Dict, List

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

SETUP_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(SETUP_DIRECTORY, "Readme.md")) as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "numpy >= 1.11",
        "tqdm",
        "optuna>=1.0.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.21.0",
        "scipy>=1.0",
        "colorlog>=4",
    ],
)

setup_requires = ["pybind11>=2.4", "requests", "setuptools_scm"]
IRSPACK_TESTING = os.environ.get("IRSPACK_TESTING", None) is not None


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        SETUP_DIRECTORY = os.path.dirname(__file__)
        target_dir = os.path.join(SETUP_DIRECTORY, self.EIGEN3_DIRNAME)
        if os.path.exists(target_dir):
            return target_dir

        download_target_dir = os.path.join(SETUP_DIRECTORY, "eigen3.zip")
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with open(download_target_dir, "wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user: Any = False):
        self.user = user

    def __str__(self) -> Any:
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "irspack._evaluator",
        ["cpp_source/evaluator.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
    ),
    Extension(
        "irspack.recommenders._ials",
        ["cpp_source/als/wrapper.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
    ),
    Extension(
        "irspack.recommenders._knn",
        ["cpp_source/knn/wrapper.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
    ),
    Extension(
        "irspack.utils._util_cpp",
        ["cpp_source/util.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler: Any, flagname: Any) -> bool:
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler: Any) -> str:
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    if IRSPACK_TESTING:
        c_opts: Dict[str, List[str]] = {
            "msvc": ["/EHsc"],
            "unix": ["-O0", "-coverage", "-g"],
        }
        l_opts: Dict[str, List[str]] = {
            "msvc": [],
            "unix": ["-coverage"],
        }
    else:
        c_opts = {
            "msvc": ["/EHsc"],
            "unix": [],
        }
        l_opts = {
            "msvc": [],
            "unix": [],
        }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


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
    setup_requires=setup_requires,
    cmdclass={"build_ext": BuildExt},
    packages=find_packages(),
    python_requires=">=3.6",
)
