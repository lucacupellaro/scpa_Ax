#!/bin/sh

# Define your array of file names
names="cage4.mtx mhda416.mtx mcfe.mtx olm1000.mtx 'adder dcop 32.mtx' west2021.mtx cavity10.mtx rdist2.mtx cant.mtx olafu.mtx 'Cube Coup dt0.mtx' 'ML Laplace.mtx' bcsstk17.mtx 'mac econ fwd500.mtx' mhd4800a.mtx 'cop20k A.mtx' raefsky2.mtx af23560.mtx lung2.mtx PR02R.mtx 'FEM 3D thermal1.mtx' thermal1.mtx thermal2.mtx 'thermomech TK.mtx' nlpkkt80.mtx webbase-1M.mtx dc1.mtx amazon0302.mtx 'af 1 k101.mtx' roadNet-PA.mtx"

# Ensure mat/ directory exists
mkdir -p mat

# Loop through each name and create the symlink
for name in $names; do
    src=$(realpath "/data/matrici/$name")
    ln -s "$src" "mat/$name"
done
