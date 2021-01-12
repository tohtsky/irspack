#!/bin/sh
pybind11-stubgen -o stubs --no-setup-py irspack.recommenders._ials
rm irspack/recommenders/_ials.pyi
echo 'm: int
n: int
from numpy import float32
' >> irspack/recommenders/_ials.pyi
cat stubs/irspack/recommenders/_ials-stubs/__init__.pyi >> irspack/recommenders/_ials.pyi
black irspack/recommenders/_ials.pyi

pybind11-stubgen -o stubs --no-setup-py irspack._evaluator
rm irspack/_evaluator.pyi
echo 'm: int
n: int
from numpy import float32
' >> irspack/_evaluator.pyi
cat stubs/irspack/_evaluator-stubs/__init__.pyi >> irspack/_evaluator.pyi
black irspack/_evaluator.pyi


pybind11-stubgen -o stubs --no-setup-py irspack.recommenders._knn
cp stubs/irspack/recommenders/_knn-stubs/__init__.pyi irspack/recommenders/_knn.pyi
black irspack/recommenders/_knn.pyi

pybind11-stubgen -o stubs --no-setup-py irspack.utils._util_cpp
rm irspack/utils/_util_cpp.pyi
echo 'from numpy import float32' >> irspack/utils/_util_cpp.pyi
cat stubs/irspack/utils/_util_cpp-stubs/__init__.pyi >> irspack/utils/_util_cpp.pyi
black irspack/utils/_util_cpp.pyi
