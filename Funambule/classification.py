# -*- coding: utf-8 -*-
# Python 3.6.2

"""
        _
    .__(.)<  (KWAK)
     \___)    
~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_eeglab, concatenate_raws
from mne.channels import read_montage
from mne.event import find_events
from mne.decoding import CSP

import numpy as np
import math

from utils import plot_covariance_matrix

from pyriemann.classification import MDM, FgMDM, TSclassifier
from pyriemann.estimation import Covariances

from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score


def main():

	# Initialize variables
	selected_channels = ["O1", "O2", "C5", "C6", "F5", "F6", "P3", "P4", "Fz", "POz", "FCz"] #selected electrodes
	montage = read_montage("standard_1005")  #pos elec head
	filter_lfreq = 0.1 
	filter_hfreq = 40.0
	
	
    # Load raw EEG data
	S2_S1_folder = "Cedric_Oliver_Live_Etats\data"
	raw_quiet1 = read_raw_eeglab(S2_S1_folder + "\Oliver_Live_BP_QuietCross1_ICA.set", montage=montage, preload=True)
	raw_quiet2 = read_raw_eeglab(S2_S1_folder + "\Oliver_Live_BP_QuietCross21_IR_ICA.set", montage=montage, preload=True)
	raw_quiet3 = read_raw_eeglab(S2_S1_folder + "\Oliver_Live_BP_QuietCross22_IR_ICA.set", montage=montage, preload=True)
	raw_anx1 = read_raw_eeglab(S2_S1_folder + "\Oliver_Live_BP_AnxCross1_IR_ICA.set", montage=montage, preload=True)
	raw_anx2 = read_raw_eeglab(S2_S1_folder + "\Oliver_Live_BP_AnxCross2_IR_ICA.set", montage=montage, preload=True)
	raw_anx3 = read_raw_eeglab(S2_S1_folder + "\Oliver_Live_BP_AnxCross3_IR_ICA.set", montage=montage, preload=True)
	raw_quiet = concatenate_raws([raw_quiet1, raw_quiet2, raw_quiet3])
	raw_anx = concatenate_raws([raw_anx1, raw_anx2, raw_anx3])   

	# Select the EEG channel :  pick only the selected channels             
	raw_quiet.pick_channels(selected_channels)
	# Apply a band-pass filter on the raw EEG signal
	raw_quiet.filter(filter_lfreq, filter_hfreq, method="iir")
	raw_anx.pick_channels(selected_channels)
	raw_anx.filter(filter_lfreq, filter_hfreq, method="iir")                
    
	# Get data from epochs in a 3D array (epoch x channel x time)
	quiet_data = raw_quiet.get_data() #(format = (nb_channels, nb_times))
	anx_data = raw_anx.get_data()
	indices = np.arange(1024, quiet_data.shape[1], 1024)   # [1024, 2048, ...]
	quiet_data = np.split(quiet_data,indices,1)  # split in 2 seconds long intervals
	quiet_data = np.array(quiet_data[:-1])    # not take last because format not correct
	
	# !!!! Faire sur chaque truc et concaténer après
	
	indices = np.arange(1024, anx_data.shape[1], 1024)
	anx_data = np.split(anx_data,indices,1)
	anx_data = np.array(anx_data[:-1])
	
	
	#supervision : first ones are 0, last ones are 1
	labels = np.array([0]*quiet_data.shape[0]+[1]*anx_data.shape[0])
	print(labels.shape)
	
	#put data together
	final_data = np.concatenate((quiet_data, anx_data), axis = 0)
	print(final_data.shape)

	cv = KFold(10, shuffle=True, random_state=42)
	# cv = LeaveOneOut()

	# CSP with Logistic Regression

	# Compute covariance matrices
	cov_data = Covariances("oas").transform(final_data)

	plot_covariance_matrix(np.mean(cov_data[:130], axis=0), np.mean(cov_data[130:], axis=0), raw_quiet.ch_names, "Mean Covariance matrices")


	lr = LogisticRegression()
	csp = CSP(n_components=4, reg="ledoit_wolf", log=True)
	csp_lr = Pipeline([("CSP", csp), ("LogisticRegression", lr)])
	scores = cross_val_score(csp_lr, final_data, labels, cv=cv, n_jobs=1)
	print("CSP with Logistic Regression Classification accuracy: %f \n" % np.mean(scores))

	# Minimum distance to mean

	mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
	scores = cross_val_score(mdm, cov_data, labels, cv=cv, n_jobs=1)
	print("MDM Classification accuracy: %f \n" % np.mean(scores))

	# Minimum distance to mean with geodesic filtering

	fg_mdm = FgMDM(metric=dict(mean='riemann', distance='riemann'))
	scores = cross_val_score(fg_mdm, cov_data, labels, cv=cv, n_jobs=1)
	print("MDM with Geodesic Filtering Classification accuracy: %f \n" % np.mean(scores))

	# Projection to the tangent space and Logistic Regression

	ts_lr = TSclassifier(clf=LogisticRegression())
	scores = cross_val_score(ts_lr, cov_data, labels, cv=cv, n_jobs=1)
	print("LR on Tangent Space Classification accuracy: %f \n\n\n" % np.mean(scores))



if __name__ == "__main__":
    main()