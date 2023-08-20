# Split training csv data into equal sized chunks
# Usage: python utils/split_csv.py --csv-folder data/data300k-with-3stars \
# --output-dir data/split-for-adv-train

import argparse
import os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv-folder', type=str, required=True)
	parser.add_argument('--output-dir', type=str, default='split_csv')
	parser.add_argument('--num-chunks', type=int, default=10)
	args = parser.parse_args()

	# create output directory if necessary
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	print(f'Will write outputs to "{args.output_dir}"')

	# load data
	train_data = pd.read_csv(f'{args.csv_folder}/train.csv')

	# Split training data into equal sized chunks
	chunk_size = int(len(train_data) / args.num_chunks)
	for i in tqdm(range(args.num_chunks)):
		chunk = train_data.iloc[i * chunk_size:(i + 1) * chunk_size]
		os.makedirs(args.output_dir + f'/{i}', exist_ok=True)
		chunk.to_csv(f'{args.output_dir}/{i}/train.csv', index=False, header=True)

		# also copy val.csv to each chunk's folder
		os.system(f'cp {args.csv_folder}/val.csv {args.output_dir}/{i}/val.csv')
	print(f'Split training data into {args.num_chunks} chunks')
