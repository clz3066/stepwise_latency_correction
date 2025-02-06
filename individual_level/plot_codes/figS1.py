import neurokit2 as nk
import os
import scipy.io as scio
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['font.family']         = 'sans-serif'   
plt.rcParams['font.sans-serif']     = 'Arial' 
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'
mm                                  = 1/25.4 # inch 和 毫米的转换

'''
https://neuropsychology.github.io/NeuroKit/functions/eeg.html
'''

file_name = os.path.join('results/ERPs/Facc_step0.mat')
eeg = scio.loadmat(file_name)['erp']
gfp = nk.eeg_gfp(eeg)
diss = nk.eeg_diss(eeg, gfp=gfp)
nk.signal_plot([gfp, diss], standardize=True)

plt.show()

