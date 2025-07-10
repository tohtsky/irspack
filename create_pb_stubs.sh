#!/bin/bash
modules=( \
"irspack.recommenders._ials_core" \
"irspack.evaluation._core_evaluator" \
"irspack.recommenders._knn" \
"irspack.utils._util_cpp"
)
for module_name in "${modules[@]}"
do
    echo "Create stub for $module_name"
    output_path="src/$(echo "${module_name}" | sed 's/\./\//g').pyi"
    python -m "nanobind.stubgen" -m "$module_name" -o "$output_path"
done
