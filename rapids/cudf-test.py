# https://towardsdatascience.com/faster-pandas-with-parallel-processing-cudf-vs-modin-f2318c594084

import pandas as pd
import time
import cudf as pd_cudf

results_groupby = []

profile_count = 5
### Read in the data with Pandas

cpu_pd = True
gpu_pd = True

def get_chunked_df_size(csv_name, delimiter, encoding):
	chunk_size = 1000000
	for run in range(0, 100):
		df_list = []
		count = 0
		size = 0
		try:
			chunkreader = pd.read_csv(csv_name,
					delimiter=delimiter,
					encoding=encoding, low_memory=False, chunksize=chunk_size)
			for df in chunkreader:
				count = count + 1
				df_list.append(df)
				size = size + count * chunk_size
				break # one big chunk is sufficient for this benchmarking
			break
		except:
			chunk_size = chunk_size / 100
			continue
	print ("Selected chunk size: ", chunk_size)

	return df_list, chunk_size

df_list, chunk_size = get_chunked_df_size("DM_ALUNO.CSV", "|", "latin-1")

if cpu_pd:
	print ("Running CPU, size = ", chunk_size)
	df = pd.concat(df_list,sort=False)
	for run in range(0, profile_count):
    
		s = time.time()
		df = df.groupby("CO_IES").size()
		e = time.time()
		results_groupby.append({"lib":"Pandas","time":float("{}".format(e-s))})
		print("Pandas Groupby Time = {}".format(e-s))

if gpu_pd:
	print ("Running GPU, size = ", chunk_size)
	df = pd_cudf.concat(df_list,sort=False)
	### Read in the data with cudf
	for run in range(0, profile_count):
		dfList = []
	    
		s = time.time()
		df = df.groupby("CO_IES").size()
		e = time.time()
		results_groupby.append({"lib":"Cudf","time":float("{}".format(e-s))})
		print("Cudf Groupby Time = {}".format(e-s))
