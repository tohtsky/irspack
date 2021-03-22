#!/bin/bash
modules=( \
"irspack.recommenders._ials" \
"irspack.evaluator._core" \
"irspack.recommenders._knn" \
"irspack.utils._util_cpp"
)
for module_name in "${modules[@]}"
do
    echo "Create stub for $module_name"
    pybind11-stubgen -o stubs --no-setup-py "$module_name"
    output_path="$(echo "${module_name}" | sed 's/\./\//g').pyi"
    input_path="stubs/$(echo "${module_name}" | sed 's/\./\//g')-stubs/__init__.pyi"
    rm "${output_path}"
    echo 'm: int
n: int
from numpy import float32
' >> "${output_path}"
    cat "${input_path}" >> "${output_path}"
    black "${output_path}"
done
