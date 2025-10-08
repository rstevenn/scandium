gcc -c ./src/data.c ./src/sc_engine.c ./src/sc_threads.c ./src/linalg.c ./src/ccbase/logs/log.c -mavx512vl -mavx512fp16 -O3 -lm
ar rsv build/scandium.a ./*.o 
del /S .\*.o