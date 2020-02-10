1. Install CUDA10.1

2. iNSTALL rapids and cuml per below

===rapids
$ conda create -n rapids python=3.6 pip
(base) $ conda activate rapids
(rapids) $ conda install -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.12 python=3.6 cudatoolkit=10.1
====cuml
conda env create -n cuml_dev python=3.6 --file=conda/environments/cuml_dev_cuda10.1.yml

==To test cudf 

>>> import cudf
>>> import cudf
>>> from cuml.cluster import DBSCAN
>>> 
>>> # Create and populate a GPU DataFrame
... gdf_float = cudf.DataFrame()
>>> gdf_float['0'] = [1.0, 2.0, 5.0]
>>> gdf_float['1'] = [4.0, 2.0, 1.0]
>>> gdf_float['2'] = [4.0, 2.0, 1.0]
>>> 
>>> # Setup and fit clusters
... dbscan_float = DBSCAN(eps=1.0, min_samples=1)
>>> dbscan_float.fit(gdf_float)
DBSCAN(eps=1.0, handle=<cuml.common.handle.Handle object at 0x7fbe5121ab88>, min_samples=1, verbose=False, max_mbytes_per_batch=0)
>>> 
>>> print(dbscan_float.labels_)
0    0
1    1
2    2
dtype: int32


