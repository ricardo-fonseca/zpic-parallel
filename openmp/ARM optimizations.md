# ARM optimizations

## A64FX processor

This is a special CPU with included HBM2. The processor is (roughly) organized as follows:

 - 48 cores total (1 thread/core)
 - 512bit SIMD vectors
 - 4 NUMA nodes

```text
$ lscpu
Architecture:        aarch64
Byte Order:          Little Endian
CPU(s):              48
On-line CPU(s) list: 0-47
Thread(s) per core:  1
Core(s) per cluster: 12
Socket(s):           -
Cluster(s):          4
NUMA node(s):        4
Vendor ID:           FUJITSU
Model:               0
Model name:          A64FX
Stepping:            0x1
BogoMIPS:            200.00
NUMA node0 CPU(s):   0-11
NUMA node1 CPU(s):   12-23
NUMA node2 CPU(s):   24-35
NUMA node3 CPU(s):   36-47
Flags:               fp asimd evtstrm sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm fcma dcpop sve
```


## Compiling for A64FX

The best solution I found was to use `gcc` 13. To compile for ARM on Deucalion using `gcc` you need to:

1. Ask for an interactive session on a production node and login to that node
2. Load the `GCC` module

```bash
[rfonseca@ln02 ~]$ local/bin/compile 
salloc: Pending job allocation 210172
salloc: job 210172 queued and waiting for resources
salloc: job 210172 has been allocated resources
salloc: Granted job allocation 210172
salloc: Waiting for resource configuration
ssh salloc: Nodes cna0001 are ready for job
[rfonseca@ln02 ~]$ ssh cna0001
Last login: Tue Nov  5 18:33:55 2024 from 10.1.0.2
[rfonseca@cna0001] module load GCC
[rfonseca@cna0001]$ gcc --version
gcc (GCC) 13.3.0
(...)
```

You can now compile the code with the usual tools. These are the options I'm using:

```
[c]     gcc -Ofast -mcpu=a64fx -Wall -std=c11
[c++]   g++ -Ofast -mcpu=a64fx -fopenmp -Wall -std=c++17 -Wno-unknown-pragmas
[ld]    g++ -Ofast -mcpu=a64fx -fopenmp -Wall -std=c++17 -Wno-unknown-pragmas 
```

For some reason, `gcc` is not detecting the architecture properly with `-march=native`

### Fujitsu compilers

While the system provides some compilers supplied by Fujitsu, these do not support setting a fixed vector width (or at least I could not find how to do it).

## NEON support

NEON (128 bit SIMD op)

## SVE support

The SVE (scalable vector extension) is an extensions to the ARM-8 architecture. In the default behavior, the vectors do not have an explicit size, and it will be up to the compiler/system to define these at compile/run time. However, because they have undefined sizes, these vectors cannot be included inside data structures.

The alternative is to define fixed width vector types like this:

```c++
typedef svfloat32_t vec_f32  __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
```

This will define and SVE vector type (`vec_f32`) that has a bit width of `__ARM_FEATURE_SVE_BITS`. This also requires that you compile the code using:

```bash
g++ -Wall -mcpu=a64fx -msve-vector-bits=<#bits>
```

Where `<#bits>` will be one of 128, 256, or 512. The `__ARM_FEATURE_SVE_BITS` macro will de defined with the same value as the `-msve-vector-bits=<#bits>` options. If the value used here does not match the system default (which is 512 bits in AF64X), you will need to add the following to the initialization section of your code:

```c++
#include <sys/prctl.h>
prctl(PR_SVE_SET_VL, __ARM_FEATURE_SVE_BITS / 8);
```

On AF64X, this will allow you to use 128, 256 or 512 bits.

_Note_: This is not A64FX specific, `-mcpu=armv8-a+sve` works on a generic ARM8-a cpu with SVE support.


## Performance

Tests were done using 1, 2 or 4 NUMA nodes inside a CPU (12, 24 or 48 cores). The code was launched using:

```bash
OMP_NUM_THREADS=12 GOMP_CPU_AFFINITY="0-11" ./zpic
OMP_NUM_THREADS=24 GOMP_CPU_AFFINITY="0-11 12-23" ./zpic
OMP_NUM_THREADS=48 GOMP_CPU_AFFINITY="0-11 12-23 24-35 36-47" ./zpic
```

Adding `OMP_WAIT_POLICY=ACTIVE` gives an extra boost.

| SIMD   |  12 cores  |  24 cores  |  48 cores |
| ------ | ---------: | ---------: | --------: |
| auto   |  0,049537  |  0,095153  |  0,175288 |
| neon   |  0,097244  |  0,185549  |  0,338017 |
| sve128 |  0,088158  |  0,168443  |  0,306903 |
| sve256 |  0,117770  |  0,224593  |  0,383820 |
| sve512 |  0,137433  |  0,261600  |  0,472390 |

## ARM Clang compilers

OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=48 OMP_PROC_BIND=true ./zpic

Performance: 0.650371 GPart/s




## Deucalion specifics

### Requesting interactive sessions

To simplify Arm development, which works better on the Arm partition, I wrote a small script to facilitate requesting an interactive session on these nodes. Once the node is made available, the script will automatically `ssh` into it.

```bash
#!/bin/bash

user="f202400003testdeucaliona"
time="0:30:00"

OPTSTRING="u:t:"

while getopts ${OPTSTRING} opt; do
    case ${opt} in
        u)
            user=${OPTARG}
            ;;
        t)
            time=${OPTARG}
            ;;
        ?)
            echo "Invalid option: -${OPTARG}" >> /dev/stderr
            exit 1
            ;;
    esac
done

tput bold
echo "Requesting interactive session..."
tput sgr0

salloc --nodes=1 --partition=dev-arm --account=${user} --time=${time} \
     ~/bin/slurm-ssh
```

