import pandas as pd
import os 
from os.path import join 
from tqdm import tqdm
import shutil
import multiprocessing


def download_file(src, dst):
	cmd = './dbxcli get {} {}'.format(src, dst)

	while True:
		ret = os.system(cmd)
		if ret != 0:
			print(f'{src} upload failed!. Retry.')
			continue
		else:
			break

def worker(inputs):
	src, dst = inputs

	download_file(src, dst)

	# unzip 
	zipfile = dst	
	oroot = dst[:-4] 
	cmd = 'unzip {} -d {}'.format(zipfile, oroot)
	ret = os.system(cmd)

	if ret == 0:
		# delete the zip 
		os.remove(zipfile)


if __name__ == '__main__':
	df = pd.read_csv('270.csv')
	hash_list = df['hash'].tolist()

	os.makedirs('MVS', exist_ok=True)

	inputs = []
	for hash in tqdm(hash_list):
		src = join('Dataset/training/', f'{hash}.zip')
		dst = join('MVS', f'{hash}.zip')

		inputs.append([src, dst])
	
	processer_num = 4 
	with multiprocessing.Pool(processer_num) as pool:
		for i, cur_diff_dict in tqdm(enumerate(pool.imap_unordered(worker, inputs), 1), total=len(inputs)):
			pass