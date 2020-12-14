from typing import Dict, List
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

install_requires = (
    [
        "numpy >= 1.11",
        "tqdm",
        "optuna>=1.0.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.21.0",
        "scipy>=1.0",
        "lightfm>=1.15",
    ],
)
setup_requires = ["pybind11>=2.4"]


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    def __str__(self):
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)
        if eigen_include_dir is None:
            setup_requires.append("requests")

        if eigen_include_dir is not None:
            return eigen_include_dir

        basedir = os.path.dirname(__file__)
        target_dir = os.path.join(basedir, self.EIGEN3_DIRNAME)
        if os.path.exists(target_dir):
            return target_dir

        download_target_dir = os.path.join(basedir, "eigen3.zip")
        import requests
        import zipfile

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

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "irspack.recommenders._rwr",
        ["cpp_source/rws.cpp"],
        include_dirs=[
            # Path to pybind11 headers
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
def has_flag(compiler, flagname):
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


def cpp_flag(compiler):
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

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": ["-march=native"],
    }
    l_opts: Dict[str, List[str]] = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
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


setup(
    name="irspack",
    version="0.2.0",
    author="Tomoki Ohtsuki",
    author_email="tomoki.otsuki129@gmail.com",
    description="Implicit feedback-based recommender system pack",
    long_description="",
    ext_modules=ext_modules,
    install_requires=install_requires,
    setup_requires=setup_requires,
    cmdclass={"build_ext": BuildExt},
    packages=[
        "irspack",
        "irspack.recommenders",
        "irspack.optimizers",
        "irspack.user_cold_start",
        "irspack.dataset",
        "irspack.dataset.movielens",
        "irspack.item_cold_start",
        "irspack.utils",
        "irspack.utils.encoders",
    ],
    package_data={"irspack": ["*.pyi"], "irspack.recommenders": ["*.pyi"]},
)
