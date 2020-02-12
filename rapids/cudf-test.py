# https://towardsdatascience.com/faster-pandas-with-parallel-processing-cudf-vs-modin-f2318c594084

import pandas as pd
import time
import cudf as pd_cudf

results_groupby = []

chunk_size = 1000000
profile_count = 5
### Read in the data with Pandas

cpu_pd = True
gpu_pd = True

if cpu_pd:
	print ("Running CPU, size = ", chunk_size)
	for run in range(0, profile_count):
		dfList = []
		count = 0
		size = 0
		try:
			chunkreader = pd.read_csv("DM_ALUNO.CSV",
					delimiter="|",
					encoding="latin-1", low_memory=False, chunksize=chunk_size)
		except:
			print ("Could not load chunk size: ", chunk_size)
			chunk_size = chunk_size / 100
			continue
		for df in chunkreader:
			count = count + 1
			dfList.append(df)
			size = size + count * chunk_size
			print ("size: ", size)
			break
		df = pd.concat(dfList,sort=False)
		    
		s = time.time()
		df = df.groupby("CO_IES").size()
		e = time.time()
		results_groupby.append({"lib":"Pandas","time":float("{}".format(e-s))})
		print("Pandas Groupby Time = {}".format(e-s))


if gpu_pd:
	print ("Running GPU, size = ", chunk_size)
	### Read in the data with cudf
	for run in range(0, profile_count):
		dfList = []
		count = 0
		size = 0
		try:
			chunkreader = pd.read_csv("DM_ALUNO.CSV",
					delimiter="|",
					encoding="latin-1", low_memory=False, chunksize=chunk_size)
		except:
			print ("Could not load chunk size: ", chunk_size)
			chunk_size = chunk_size / 100
			continue
		for df in chunkreader:
			count = count + 1
			dfList.append(df)
			size = size + count * chunk_size
			print ("size: ", size)
			break
		df = pd_cudf.concat(dfList,sort=False)
	    
		s = time.time()
		df = df.groupby("CO_IES").size()
		e = time.time()
		results_groupby.append({"lib":"Cudf","time":float("{}".format(e-s))})
		print("Cudf Groupby Time = {}".format(e-s))
