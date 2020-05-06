import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("Metrics_GAN/Metrics_LOGAN.csv")
#df2 = pd.read_csv("Metrics_GAN/Metrics_LOGAN.csv")
epoch_start = 0
epoch_end = 4951
epochs = np.arange(epoch_start, epoch_end, 100)
x = epochs[:(len(df))]

fig, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].plot(x,df['FID'], '-+', color = 'red', label = 'LOGAN')
#ax[0].plot(x,df2['FID'], '-+', color = 'blue', label = 'DCGAN with LOGAN')
ax[0].set_title('FID Score of generated images')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("FID")
ax[0].legend(loc = 'upper left')
ax[0].grid()
ax[1].plot(x,df['Inception Score'], '-+',color = 'red', label = 'LOGAN')
#ax[1].plot(x,df2['Inception Score'], '-+',color = 'blue', label = 'DCGAN')
ax[1].set_title('Inception Score of generated images')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IS")
ax[1].legend(loc = 'upper left')
ax[1].grid()
plt.show()

#plt.savefig("DCGAN_original.png")
