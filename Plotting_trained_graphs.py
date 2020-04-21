import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("Metrics_GAN/Metrics_GAN_ResNet_3900.csv")
epoch_start = 0
epoch_end = 4000
epochs = np.arange(epoch_start, epoch_end, 100)
x = epochs[:(len(df))]

fig, ax = plt.subplots(1,2,figsize = (14,5))
ax[0].plot(x,df['FID'], '-ok', color = 'blue', label = 'FID ResNet34')
ax[0].set_title('FID Score of generated images')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("FID")
ax[0].legend()

ax[1].plot(x,df['Inception Score'], '-ok',color = 'blue', label = 'IS ResNet34')
ax[1].set_title('Inception Score of generated images')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IS")
ax[1].legend()

plt.show()

plt.savefig("DCGAN_original.png")
