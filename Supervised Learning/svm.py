# create synthetic data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)

# plot zero lines between classes
import numpy as np
# xfit = np.linspace(-1, 3.5)
import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
# plt.plot([0.6], [2.1], 'x', color='black', markeredgewidth=4, markersize=12)
# for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
#     plt.plot(xfit, m * xfit + b, '-k')
# plt.xlim(-1, 3.5)
# plt.show()

# plot margins between classes
# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
# for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
#     yfit = m * xfit + b
#     plt.plot(xfit, yfit, '-k')
#     plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
#     plt.xlim(-1, 3.5)
# plt.show()

# data training
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

# 2d svc function
def decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    if plot_support:
        # Added a label here so the user knows what the giant empty circles represent
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

# data visualization
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap='summer', edgecolors='k', alpha=0.8)
decision_function(model)
handles, _ = scatter.legend_elements()
ax = plt.gca()
handles_ax, labels_ax = ax.get_legend_handles_labels()
plt.legend(
    handles + handles_ax, 
    ['Class 0', 'Class 1'] + labels_ax, 
    loc='upper right', 
    frameon=True
)
plt.title('SVM: Classifier & Margins', weight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Model-Image/svm.png', dpi=300)