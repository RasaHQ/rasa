#!/bin/bash

# to enable remote execution
cd "$(dirname "$0")"
cd ..

source env/bin/activate

make test

echo "!!!!!! THERE SHOULD BE 1044 TESTS PASSED !!!!!!"
echo "!!! SMALLER NUMBER IS CONSIDERED AS FAILURE !!!"
