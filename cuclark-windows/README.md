# cuClark on Windows


First attempt after fixing minor issues - commit id 6ca2168047446acda8cec98cb7cd7e3f5f2ef6e1

Download the example db per instructions

Build and run cuclark from https://github.com/prabindh/cuclark

```
<>example_db><>\cuclark\windows\cuclark\x64\Release\cuclark.exe -T .\targets.txt -D .\custom_0_canonical -O .\dataset\HiSeq_accuracy.fa -R .\dataset\HiSeq_accuracy_output.fa -d 1
CuCLARK version 1.1 (Copyright 2016 Robin Kobus, rkobus@students.uni-mainz.de)
Based on CLARK version 1.1.3 (UCR CS&E. Copyright 2013-2016 Rachid Ounit, rouni001@cs.ucr.edu)
Starting the creation of the database of targets specific 31-mers from input files...
 Progress report: (22/22)    39365954 nt read in total.
Mother Hashtable successfully built. 38779811 31-mers stored.
Hashtable sorting done: maximum number of collisions: 4
Removal of common k-mers done: 38774466 specific 31-mers found.
Creating database in disk...
38774466 31-mers successfully stored in database.
Checking for CUDA devices: 1 device(s) found.
Device 0 = Quadro M1000M
Using 1 CUDA devices as requested.
Loading database [.\custom_0_canonical//db_central_k31_t9_s1610612741_m0.tsk.*] (s=1)...
Total DB size in RAM:   6.675 GB
Total device memory:    1.365 GB (400 MB reserved)
Requiring 5 loop(s).
DB loaded in RAM.
Processing file '.\dataset\HiSeq_accuracy.fa' in 1 batches using 1 CPU thread(s).
```

After fixing more bugs, able to run cuclark test, but the output csv has all lines with confidence 0

```
CuCLARK version 1.1 (Copyright 2016 Robin Kobus, rkobus@students.uni-mainz.de)
Based on CLARK version 1.1.3 (UCR CS&E. Copyright 2013-2016 Rachid Ounit, rouni001@cs.ucr.edu)
Checking for CUDA devices: 1 device(s) found.
Device 0 = Quadro M1000M
Using 1 CUDA devices as requested.
Loading database [.\custom_0_canonical//db_central_k31_t9_s4_m0.tsk.*] (s=1)...
Total DB size in RAM:   6.83 GB
Total device memory:    1.365 GB (400 MB reserved)
Requiring 6 loop(s).
DB loaded in RAM.
Processing file '.\dataset\HiSeq_accuracy.fa' in 1 batches using 1 CPU thread(s).
Batch 0: AVG read length 93, Estimated # containers per read: 14
Estimated # containers per read: 14
Batch 0  # Containers: 128840 (AVG # per read: 12.884).
Host 0 Batch 0   CUDA start.    Read data size: 0 MB    Result data size: 0.05 MB
Batch 0 scheduled. Objects: 10000 Size: 0.297 MB
Writing results...
Done.
CUERR 'an illegal memory access was encountered' in <>/cuclark/src/CuClarkDB.cu, line 287
```