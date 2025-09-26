gcc ./src/generate_tests.c ./ccbase/logs/log.c -o ./build/gen_test.exe -lm -I ./ccbase -I ./src 
.\build\gen_test.exe
gcc ./src/test.c ./src/data.c ./src/linalg.c ./ccbase/logs/log.c  -o ./build/test.exe  -lm -I ./ccbase -I ./src
.\build\test.exe