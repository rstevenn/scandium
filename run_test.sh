#!/bin/sh
set -ex
gcc ./src/generate_tests.c ./src/ccbase/logs/log.c -o ./build/gen_test -lm -I ./ccbase -I ./src
./build/gen_test
gcc ./src/test.c ./src/data.c ./src/linalg.c ./src/ccbase/logs/log.c -march=x86-64 -mavx512vl -mavx512fp16 ./src/sc_engine.c ./src/sc_threads.c  -o ./build/test  -lm -I ./ccbase -I ./src
./build/test
