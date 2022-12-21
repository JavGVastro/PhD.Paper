from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

figfile = "bright-to-vel-fluct.pdf"
sns.set_context("talk")
sns.set_color_codes()
fig, ax = plt.subplots()

alpha = np.logspace(0.0, 1.6, 500)

def plot_and_label(ax, x, y, label, pad="  "):
    [line] = ax.plot(x, y)
    ax.text(x[-1], y[-1], pad + label,
            color=line.get_color(), ha="left", va="center",
            )

plot_and_label(
    ax,
    alpha,
    np.sqrt(alpha)  / (1 + alpha),
    label=r"$\sigma_{\mathrm{los}} / \beta$",
)
extra_sig = 0.25
ax.plot(
    alpha,
    np.hypot(np.sqrt(alpha)  / (1 + alpha), extra_sig),
    linestyle="dashed",
    color=ax.lines[-1].get_color(),
)
plot_and_label(
    ax,
    alpha,
    0.5 * np.abs(alpha - 1)  / (1 + alpha),
    label=r"$\sigma_{\mathrm{pos}} / \beta$",
)
plot_and_label(
    ax,
    alpha,
    np.abs(alpha - 1)  / (1 + alpha),
    label=r"$\sigma_S$",
)
gamma = alpha + 1/alpha
plot_and_label(
    ax,
    alpha,
    np.sqrt(3 * gamma ** 2 - 4 * (1 + gamma)) / (2 + gamma),
    label=r"$\sigma_E$",
)
ax.set(
    xscale="log",
    xlabel=r"Emissivity contrast, $\alpha$",
)
ax.set_xticks([1, 3, 10, 30])
ax.set_xticklabels(["1", "3", "10", "30"])
ax.set_xlim(1, None)
ax.set_yticks([0.0, 0.5, 1.0, 1.5])
ax.set_yticklabels(["", "0.5", "1.0", "1.5"])
ax.set_ylim(0, None)
sns.despine()
fig.savefig(figfile, bbox_inches="tight")

print(figfile, end="")
