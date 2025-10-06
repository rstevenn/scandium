gcc -c ./src/data.c ./src/sc_engine.c ./src/sc_threads.c ./src/linalg.c ./ccbase/logs/log.c -O3 -lm -I ./ccbase -I ./src
ar rsv build/scandium.a ./*.o 
del /S .\*.o