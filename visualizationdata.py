import matplotlib.pyplot as plt
import numpy as np

arr = np.load(r"C:\Users\kirit\OneDrive\Documents\Semester 2\Introduction to AIML\3D_Localization_real_data\dataset_mird\mird_000000.npy")
plt.imshow(arr , aspect = "auto" , origin = "lower" , cmap = "coolwarm")
plt.colorbar(label = "IPD (deg)")
plt.xlabel("Time Frame")
plt.ylabel("Frequency bin")
plt.title("sample_000001.npy")
plt.show()