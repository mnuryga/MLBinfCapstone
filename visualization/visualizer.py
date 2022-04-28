import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from scipy import stats

def normalize(preds):
	preds[:, 0] = (preds[:, 0]-np.min(preds[:, 0]))/np.ptp(preds[:, 0])
	preds[:, 1] = (preds[:, 1]-np.min(preds[:, 1]))/np.ptp(preds[:, 1])
	preds[:, 2] = (preds[:, 2]-np.min(preds[:, 2]))/np.ptp(preds[:, 2])
	return preds

def compute_nearest_attachment(preds)
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

def visualize(preds, labels):
	t = np.pi/3+np.pi
	preds[:, 0] = preds[:, 0]*np.cos(t) - preds[:, 1]*np.sin(t)
	preds[:, 1] = preds[:, 0]*np.sin(t) + preds[:, 1]*np.cos(t)

	preds = normalize(preds)
	order = compute_nearest_attachment(preds)
	preds[:, 0] -= preds[:, 2]/10
	preds[:, 1] -= preds[:, 2]/10
	order[:, 0] -= order[:, 2]/10
	order[:, 1] -= order[:, 2]/10

	ax = plt.gca(projection="3d")
	ax.scatter(preds[:, 0],preds[:, 1],preds[:, 2], c='b', s=10)

	ax.plot(order[:, 0],order[:, 1],order[:, 2], color='r', lw=1)

	t = np.pi/3+np.pi
	preds[:, 0] = preds[:, 0]*np.cos(t) - preds[:, 1]*np.sin(t)
	preds[:, 1] = preds[:, 0]*np.sin(t) + preds[:, 1]*np.cos(t)

	labels = normalize(labels)

	ax = plt.gca(projection="3d")
	ax.scatter(preds[:, 0],preds[:, 1],preds[:, 2], c='green', s=10)

	ax.plot(labels[:, 0],labels[:, 1],labels[:, 2], color='pink', lw=1)

	plt.show()
	plt.clf()

def loss_processing(losses, lens):
	plt.boxplot(losses, notch = True)
	plt.ylabel('Test Loss per C_alpha')
	plt.title('Distribution of Test Losses')
	plt.savefig('../media/loss_boxplot.png', dpi = 250)

	plt.clf()
	slope, intercept, r_value, p_value, std_err = stats.linregress(lens, losses)
	plt.plot(lens,losses)
	plt.plot(lens, lens*slope+intercept, label = f'Regression Line, r^2={r_value**2:.4f}')
	plt.xlabel('Sequence Length')
	plt.ylabel('Test Loss per C_alpha')
	plt.title('Test Loss vs. Sequence Length')
	plt.legend()
	plt.savefig('../media/loss_seqs.png', dpi = 250)

def main():
	losses = np.load('losses.npy')
	lens = np.load('s_len.npy')
	# visualize()
	loss_processing(losses, lens)

if __name__ == '__main__':
	main()