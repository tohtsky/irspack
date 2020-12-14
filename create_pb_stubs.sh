#!/bin/sh
pybind11-stubgen -o stubs --no-setup-py irspack.recommenders._knn
mv stubs/irspack/recommenders/_ials-stubs/__init__.pyi irspack/recommenders/_ials.pyi

pybind11-stubgen -o stubs --no-setup-py irspack.recommenders._ials
mv stubs/irspack/recommenders/_knn-stubs/__init__.pyi irspack/recommenders/_knn.pyi

rm -rf stubs