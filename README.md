# Perceptron Branch Predictor
* Implement in gem5


## Execution 
* git clone https://gem5.googlesource.com/public/gem5
* cd gem5
* scons build/X86/gem5.opt -j <number of threads>
* add `perceptron.cc`, `perceptron.hh`, `BranchPredictor.py` in `./gem5/src/cpu/pred`
```bash
build/X86/gem5.opt configs/example/l3se.py --cpu-type X86O3CPU --bp-type PerceptronBP  --caches --l1i_size 32kB -I 10000000 --l1i_assoc 8 --l1d_size 32kB --l1d_assoc 8 --l2cache --l2_size 512kB --l2_assoc 8 --l3cache --l3_size 16MB --l3_assoc 16  -c {Your Work Load}
```

