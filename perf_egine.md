### Single thread no SIMD, no thread pool
```
Stress test took 3.173190 seconds per iteration.
Engine speed: 53.57 Mop/s
```

### Multi thread for [ element_wise_op, scalar_op, reduce ] no SIMD, no thread pool

```
Stress test took 1.470380 seconds per iteration.
Engine speed: 115.62 Mop/s
```

### Multi thread for [ element_wise_op, scalar_op, reduce, map, map_args ] no SIMD, no thread pool
```
Stress test took 0.363250 seconds per iteration.
Engine speed: 468.00 Mop/s
```