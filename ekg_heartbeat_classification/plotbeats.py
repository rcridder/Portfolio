"""
All code written by Rose Ridder and Holden Bridge
All data was taken from https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm via Kaggle
"""

from accessFileTest import getData
import matplotlib.pyplot as plt
import numpy as np

scratchfilepath = "../../../../scratch/rridder1/" # edit for your username
train_filename = "mitbih_train.csv"
test_filename = "mitbih_test.csv"

names = ["Normal", "Supraventricular premature beat",
"Premature ventricular contraction", "Fusion of ventricular and normal beat",
"Unclassifiable Beat"]

#dat = getData(scratchfilepath+train_filename)[67471:]
dat = getData(scratchfilepath+train_filename)[16498:]

num_plots = 8
store_dat = np.zeros(shape = (5*num_plots, dat.shape[1]))
plots = np.zeros(5, dtype = int)
for row in dat:
    if np.array_equal(plots, np.ones(5)*num_plots): break
    typ = int(row[-1])
    if plots[typ] < num_plots:
        plt.subplot(num_plots, 5, 5*plots[typ]+(typ+1))
        data = row[:-1]
        plt.plot(range(len(data)), data)
        if plots[typ] == 0:
            plt.title(names[typ])
        store_dat[typ*num_plots+plots[typ]] = row
        plots[typ]+=1
np.savetxt("save_data.csv", store_dat, delimiter = ",")

plt.show()
