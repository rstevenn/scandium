call .\build_lib.bat
gcc ./src/perfs.c ./build/scandium.a -O3 -o ./build/perf.exe 
.\build\perf.exe