import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_2d_probability_heatmap(x, y, bins=50, cmap='viridis',savepath=''):
    # Count in each 2D bin
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Convert counts to probability
    P = H / np.sum(H)*100

    plt.figure(figsize=(3.2, 2.5))
    plt.pcolormesh(
        xedges,
        yedges,
        P.T,
        cmap=cmap,
        shading='auto',
         vmin=0, vmax=0.5,
    )

    plt.colorbar(label='Probability (%)')
    plt.ylabel('Axis amplitudes')
    plt.xlabel('State timeseries value')
    # plt.title('2D probability heatmap')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    return P, xedges, yedges

if __name__ == '__main__':
    sample_dir = Path(__file__).resolve().parent
    form_list = np.load(sample_dir / 'state_expression_list.npy')
    axe_list = np.load(sample_dir / 'axe_amplitude_list.npy')
    for i in range(form_list.shape[1]):
        plot_2d_probability_heatmap(form_list[:, i], axe_list[:, i], bins=50, cmap='hot', savepath=f'axis_state_heatmap_{i}.png')
