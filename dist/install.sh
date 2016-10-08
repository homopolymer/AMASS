#!/usr/bin/env bash

# path to g++ 4.9 compiler
# modify it if it cannot point to the right path in your system
gxx49=$(which g++-4.9)

mkdir -p bin

$gxx49 -std=c++11 -o bin/StrainCall StrainCall/*.cpp
cp scripts/amass.py bin/amass.py

echo done!
