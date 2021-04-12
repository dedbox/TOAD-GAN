import numpy as np
import matplotlib.pyplot as plt
import matplotlib

loss24 = []
avg_loss24 = []
for L in range(64):
    data = np.load(f'vae-sweep-24-L/vae-lvl_1-1-24-{L+1}.npz')
    loss24.append(data['loss'])
    avg_loss24.append(data['avg_loss'])

# show = [7, 8, 13, 14, 15, 16] + list(range(17, 65))

# loss24 = np.array(loss24)
# avg_loss24 = np.array(avg_loss24)

plt.figure(figsize=(10, 5))
plt.title("VAE Sweep lvl_1-1 (all), H=24, $1 \leq L \leq 64$")
for L, data in enumerate(avg_loss24):
    # if L+1 in show:
        plt.plot(data[:10], label=L+1)
plt.legend(ncol=7)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f'vae-sweep-24-L-all.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()

plt.close()

plt.figure(figsize=(5, 5))
plt.title("VAE Sweep lvl_1-1 (best), H=24, $1 \leq L \leq 64$")
plt.plot(avg_loss24[6], label=7)
plt.plot(avg_loss24[7], label=8)
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(f'vae-sweep-24-L-best.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()

################################################################################

loss7 = []
avg_loss7 = []
for H in range(50):
    data = np.load(f'vae-sweep-H-7/vae-lvl_1-1-{H+1}-7.npz')
    loss7.append(data['loss'])
    avg_loss7.append(data['avg_loss'])

avg_loss7 = np.array(avg_loss7)

show = [42, 49]

# colors = [matplotlib.cm.prism(x) for x in np.linspace(0, 1, 50)]

plt.figure(figsize=(10, 5))
plt.title("VAE Sweep lvl_1-1 (all), L=7, $1 \leq H \leq 50$")
for H, data in enumerate(avg_loss7):
    # if H+1 in show:
        plt.plot(data[:10], label=H+1)
# data = np.gradient(np.array(avg_loss7), axis=1)
# plt.plot(data)
# for i, color in enumerate(colors):
#     if i >= 35:
#         plt.plot(np.log(avg_loss7[i, 7:10]), color=color, label=i+1)
# plt.plot(avg_loss7.T, color=colors)
plt.legend(ncol=7)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f'vae-sweep-H-7-all.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()

plt.close()

plt.figure(figsize=(5, 5))
plt.title("VAE Sweep lvl_1-1 (best), L=7, $1 \leq L \leq 64$")
plt.plot(avg_loss7[41], label=42)
plt.plot(avg_loss7[48], label=49)
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(f'vae-sweep-H-7-best.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()
