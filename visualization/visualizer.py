import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from scipy import stats

def normalize(preds):
	'''
	normalizes the coordinates to [0,1]
	'''
	preds[:, 0] = (preds[:, 0]-np.min(preds[:, 0]))/np.ptp(preds[:, 0])
	preds[:, 1] = (preds[:, 1]-np.min(preds[:, 1]))/np.ptp(preds[:, 1])
	preds[:, 2] = (preds[:, 2]-np.min(preds[:, 2]))/np.ptp(preds[:, 2])
	return preds

def compute_nearest_attachment(preds):
	'''
	For a set of coordinates, picks a starting point and
	constructs an array of coordinates where each element
	is ordered by euclidean distance to adjacent points
	'''
	order = np.zeros_like(preds)
	order[0] = preds[0]
	remaining = preds.tolist()
	for i in range(1, len(order)):
		current = order[i-1]
		d = np.zeros(len(remaining))
		for j, r in enumerate(remaining):
			d[j] = ((current[0] - r[0])**2 + (current[1] - r[1])**2 + (current[2] - r[2])**2)**0.5
		idx = np.argmin(d)
		order[i] = np.array(remaining[idx])
		del remaining[idx]

	return order

def visualize(preds, labels, realign = False):
	'''
	given 2 coordinate arrays, plot them in 3d=space
	and rearrange them to overlay on each other
	'''
	t = np.pi/3+np.pi
	preds[:, 0] = preds[:, 0]*np.cos(t) - preds[:, 1]*np.sin(t)
	preds[:, 1] = preds[:, 0]*np.sin(t) + preds[:, 1]*np.cos(t)

	preds = normalize(preds)
	# preds[:, 2] = 1-preds[:, 2]
	if realign:
		order = compute_nearest_attachment(preds)
	else:
		order = preds
	preds[:, 0] -= preds[:, 2]/20
	preds[:, 1] -= preds[:, 2]/20
	order[:, 0] -= order[:, 2]/20
	order[:, 1] -= order[:, 2]/20

	ax = plt.gca(projection="3d")
	ax.scatter(preds[:, 0],preds[:, 1],preds[:, 2], c='b', s=20)

	ax.plot(order[:, 0],order[:, 1],order[:, 2], color='r', lw=1, label = 'predicted')

	t = np.pi/3+np.pi
	labels[:, 0] = labels[:, 0]*np.cos(t) - labels[:, 1]*np.sin(t)
	labels[:, 1] = labels[:, 0]*np.sin(t) + labels[:, 1]*np.cos(t)

	labels = normalize(labels)

	ax = plt.gca(projection="3d")
	ax.scatter(labels[:, 0],labels[:, 1],labels[:, 2], c='green', s=20)

	ax.plot(labels[:, 0],labels[:, 1],labels[:, 2], color='black', lw=1, label = 'expected')
	ax.legend()

	plt.show()
	plt.clf()

def loss_processing(losses, lens):
	'''
	plots of basic loss data
	'''
	plt.boxplot(losses, notch = True)
	plt.ylabel('Test Loss per Sequence')
	plt.title('Distribution of Test Losses')
	plt.savefig('../media/loss_boxplot.png', dpi = 250)

	plt.clf()
	slope, intercept, r_value, p_value, std_err = stats.linregress(lens, losses)
	plt.plot(lens,losses)
	plt.plot(lens, lens*slope+intercept, label = f'Regression Line, r^2={r_value**2:.4f}')
	plt.xlabel('Sequence Length')
	plt.ylabel('Test Loss per Sequence')
	plt.title('Test Loss vs. Sequence Length')
	plt.legend()
	plt.savefig('../media/loss_seqs.png', dpi = 250)

def main():
	coords = np.load('all_coords.npy')[:, 1:-1]
	preds = np.load('all_preds.npy')[:, 1:-1]
	losses = np.load('losses.npy')
	lens = np.load('s_len.npy')
	best_idx = np.argmin(losses)
	med_idx = 26
	worst_idx = np.argmax(losses)
	print(losses[best_idx])
	print(losses[med_idx])
	print(losses[worst_idx])
	visualize(preds[worst_idx, 1:lens[worst_idx]-1], coords[worst_idx, 1:lens[worst_idx]-1], realign = True)
	loss_processing(losses, lens)

if __name__ == '__main__':
	main()