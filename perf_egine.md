### Single thread no SIMD, no thread pool
```
Stress test took 3.173190 seconds per iteration.
Engine speed: 53.57 Mop/s
```

### Multi thread for [ element_wise_op, scalar_op, reduce ] no SIMD, no thread pool

```
Stress test took 1.428570 seconds per iteration.
Engine speed: 133.00 Mop/s
```

### Multi thread for [ element_wise_op, scalar_op, reduce, map, map_args, dot ] no SIMD, no thread pool
```
Stress test took 0.332030 seconds per iteration.
Engine speed: 572.24 Mop/s
```