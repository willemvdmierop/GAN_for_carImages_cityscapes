import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("Metrics_GAN/Metrics_ResNet18_cropped_withGradclip.csv")
epoch_start = 0
epoch_end = 4951
epochs = np.arange(epoch_start, epoch_end, 50)
x = epochs[:(len(df))]

fig, ax = plt.subplots(1,2,figsize = (14,5))
ax[0].plot(x,df['FID'], '-+', color = 'red', label = 'FID ResNet18 with gradient clipping')
ax[0].set_title('FID Score of generated images')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("FID")
ax[0].legend()
ax[0].grid()
ax[1].plot(x,df['Inception Score'], '-+',color = 'red', label = 'IS ResNet18 with gradient clipping')
ax[1].set_title('Inception Score of generated images')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IS")
ax[1].legend()
ax[1].grid()
plt.show()

#plt.savefig("DCGAN_original.png")
