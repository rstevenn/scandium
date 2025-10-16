gcc ./src/generate_tests.c ./src/ccbase/logs/log.c -o ./build/gen_test.exe -lm
.\build\gen_test.exe
gcc ./src/test.c ./src/data.c ./src/linalg.c ./src/ccbase/logs/log.c ./src/sc_engine.c ./src/sc_threads.c -mavx -ggdb -o ./build/test  -lm
.\build\test.exe