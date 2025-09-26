#!/bin/sh
set -ex
gcc ./src/generate_tests.c ./ccbase/logs/log.c -o ./build/gen_test -lm -I ./ccbase -I ./src
./build/gen_test
gcc ./src/test.c ./src/data.c ./src/linalg.c ./ccbase/logs/log.c  -o ./build/test  -lm -I ./ccbase -I ./src
./build/test