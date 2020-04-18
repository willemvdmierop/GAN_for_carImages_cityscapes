import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("Metrics_DCGAN_batch.csv")
epoch_start = 0
epoch_end = 2100
epochs = np.arange(epoch_start, epoch_end, 100)
x = epochs[:(len(df))]

fig, ax = plt.subplots(2,1,figsize = (10,10))
ax[0].plot(x,df['FID'])
ax[0].set_title('FID Score of generated images')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("FID")

ax[1].plot(x,df['Inception Score'])
ax[1].set_title('Inception Score of generated images')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IS")


plt.show()

plt.savefig("DCGAN_original.png")
