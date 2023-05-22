import os
import sys


def main():
	# Base experiment directory
	base_dir = sys.argv[1]
	assert(os.path.isdir(base_dir))
	K = None
	if len(sys.argv) == 3:
		# Num of neighbors
		K = sys.argv[2]
	# Get directories for all PartNet categories
	if K is not None:
		temp = '-k{}-'.format(K)
		experiments_dir = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if temp in f]
	else:
		experiments_dir = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
	experiments_dir = sorted(experiments_dir)

	# Get shape and part IoU
	postfix = "evaluation"
	part_iou, shape_iou = [], []
	for experiment in experiments_dir:
		for f in os.listdir(experiment):
			if f.endswith(postfix):
				results_log = os.path.join(experiment, f, "results", "results_log.txt")
				with open(results_log, 'r') as fin:
					for line in fin:
						line = line.strip().split()
						if line[0].lower() == 'part':
							part_iou.append(float(line[-1]))
						if line[0].lower() == 'shape':
							shape_iou.append(float(line[-1]))

	# Print results
	print("PART IOU:")
	print("---------")
	print(part_iou)
	buf = '=SPLIT("'
	for i in range(len(part_iou) - 1):
		buf += str(part_iou[i]) + ','
	buf += str(part_iou[-1]) + '", ",")'
	print('{}' .format(buf))

	print("SHAPE IOU:")
	print("----------")
	print(shape_iou)
	buf = '=SPLIT("'
	for i in range(len(shape_iou) - 1):
		buf += str(shape_iou[i]) + ','
	buf += str(shape_iou[-1]) + '", ",")'
	print('{}' .format(buf))


if __name__ == '__main__':
	main()
