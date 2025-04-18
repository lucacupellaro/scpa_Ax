#!/bin/sh

# Define your array of file names
names="cage4.mtx mhda416.mtx mcfe.mtx cant.mtx amazon0302.mtx"
#names="cage4.mtx mhda416.mtx mcfe.mtx olm1000.mtx adder_dcop_32.mtx west2021.mtx cavity10.mtx rdist2.mtx cant.mtx olafu.mtx Cube_Coup_dt0.mtx ML_Laplace.mtx bcsstk17.mtx mac_econ_fwd500.mtx mhd4800a.mtx cop20k_A.mtx raefsky2.mtx af23560.mtx lung2.mtx PR02R.mtx FEM_3D_thermal1.mtx thermal1.mtx thermal2.mtx thermomech_TK.mtx nlpkkt80.mtx webbase-1M.mtx dc1.mtx amazon0302.mtx af_1_k101.mtx roadNet-PA.mtx"
# Ensure mat/ directory exists
mkdir -p mat

# Loop through each name and create the symlink
for name in $names; do
    src=$(realpath "/data/matrici/$name")
    ln -s "$src" "mat/$name"
    echo "$name" # Print each name on a new line
done
