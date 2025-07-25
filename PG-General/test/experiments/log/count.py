import numpy as np
import matplotlib.pyplot as plt

# Sample sparse data for plotting
nx, ny = 200, 200

x1 = np.zeros(nx + ny)
x1[47] = 0.05670878 
x1[362] = 0.0583163

x2 = np.zeros(nx + ny)
x2[3] = 0.0346289
x2[25] = -0.02353222
x2[352] = -0.01957639
x2[392] = 0.00678256
x2[393] = 0.03306209

x3 = np.zeros(nx + ny)
x3[1] = -2.34345558e-2
x3[16] = -2.82129718e-2
x3[358] = 4.46061493e-3
x3[364] = 7.7839732e-3
x3[375] = 1.41365466e-2
x3[391] = -1.0251847e-2
x3[395] = -1.64947615e-2
x3[396] = -6.58961873e-3
x3[399] = -3.50250691e-5

x4 = np.zeros(nx + ny)
x4[3] = -0.0083173
x4[19] = -0.00556405
x4[34] = 0.01120129
x4[41] = 0.03028209
x4[43] = 0.00728848
x4[361] = 0.0085047
x4[374] = 0.00269642
x4[380] = -0.02181036
x4[397] = -0.02302685

# Indices
idx = np.arange(1, nx + ny + 1)

# Create figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# def plot_stem(ax, x, label):
#     ax.stem(idx, x, basefmt='C1-')
#     ax.set_xlim([1, len(x)])
#     ax.set_xticks([0, 50, 350, 400])
#     ax.axvline(x=50, color='gray', linestyle='--', linewidth=1)
#     ax.axvline(x=350, color='gray', linestyle='--', linewidth=1)
#     ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
#     ax.set_title(label, fontsize=12)

def plot_stem(ax, x, label):
    markerline, stemlines, baseline = ax.stem(idx, x, basefmt='k-', linefmt='b-', markerfmt='bo')
    
    # Optional: make all stem lines the same color manually
    plt.setp(stemlines, color='steelblue', linewidth=2)
    plt.setp(markerline, color='steelblue', markersize=7)  # Orange markers
    plt.setp(baseline, color='orange', linewidth=4)  # Black baseline
    
    ax.set_xlim([1, len(x)])
    ax.set_xticks([0, 50, 350, 400])
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=350, color='gray', linestyle='--', linewidth=2)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax.set_title(label, fontsize=12)

plot_stem(axs[0][0], x1, r"(a) $\lambda = 1e\text{-}2$")
plot_stem(axs[0][1], x2, r"(b) $\lambda = 5e\text{-}3$")
plot_stem(axs[1][0], x3, r"(c) $\lambda = 2e\text{-}3$")
plot_stem(axs[1][1], x4, r"(d) $\lambda = 1e\text{-}3$")

fig.suptitle(r"Sparse Canonical Correlation Analysis (SCCA) Results ($n_x=n_y=200$)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
plt.savefig("scca.pdf")
plt.show()




from brokenaxes import brokenaxes

fig = plt.figure(figsize=(12, 7))
axs = []

# Coordinates for subplot positioning
positions = [(0.05, 0.55), (0.55, 0.55), (0.05, 0.05), (0.55, 0.05)]

for i, (x, label) in enumerate(zip([x1, x2, x3, x4],
                                   [r"(a) $\lambda = 1e\text{-}2$",
                                    r"(b) $\lambda = 5e\text{-}3$",
                                    r"(c) $\lambda = 2e\text{-}3$",
                                    r"(d) $\lambda = 1e\text{-}3$"])):
    # Create broken axis: show 0–60 and 340–400
    bax = brokenaxes(xlims=((0, 60), (340, 400)), hspace=0.05, subplot_spec=fig.add_subplot(2, 2, i + 1))
    markerline, stemlines, baseline = bax.stem(idx, x, basefmt='k-', linefmt='b-', markerfmt='bo')
    plt.setp(stemlines, color='steelblue', linewidth=2)
    plt.setp(markerline, color='steelblue', markersize=7)
    plt.setp(baseline, color='orange', linewidth=4)
    bax.set_title(label, fontsize=12)
    axs.append(bax)

fig.suptitle(r"Sparse Canonical Correlation Analysis (SCCA) Results ($n_x=n_y=200$)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("scca.pdf")
plt.show()
