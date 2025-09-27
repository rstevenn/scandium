gcc ./src/perfs.c ./src/data.c ./src/sc_engine.c ./src/sc_threads.c ./src/linalg.c ./ccbase/logs/log.c -O3 -o ./build/perf.exe  -lm -I ./ccbase -I ./src
.\build\perf.exe