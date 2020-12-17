#!/bin/sh
pybind11-stubgen -o stubs --no-setup-py irspack.recommenders._ials
cp stubs/irspack/recommenders/_ials-stubs/__init__.pyi irspack/recommenders/_ials.pyi

pybind11-stubgen -o stubs --no-setup-py irspack.recommenders._knn
cp stubs/irspack/recommenders/_knn-stubs/__init__.pyi irspack/recommenders/_knn.pyi

pybind11-stubgen -o stubs --no-setup-py irspack.utils._util_cpp
cp stubs/irspack/utils/_util_cpp-stubs/__init__.pyi irspack/utils/_util_cpp.pyi

pybind11-stubgen -o stubs --no-setup-py irspack._evaluator
cp stubs/irspack/_evaluator-stubs/__init__.pyi irspack/_evaluator.pyi
