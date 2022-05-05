'''Written in May 2022 to calculate kmax parameter for Higuchi's Fractal Dimension'''

import antropy as ant
import numpy as np
import stochastic.processes.noise as sn
import pandas as pd
from prepare_files import hof_load_files, hof_extract_brainstate_REM_wake
from filter import hof_filter
from constants import baseline_recording_dictionary, start_times_baseline
from statistics import mean

directory_path = '/home/melissa/preprocessing/numpyformat_baseline' 
brainstate_number = [1]
kmax_range = list(range(2, 100))
channel_number = [7]
for brainstate in brainstate_number:
    higuchi_df = []
    for animal in baseline_recording_dictionary['animal_two_brainstates']:
        data_1, data_2, brain_state_1, brain_state_2 = hof_load_files(directory_path, animal, start_times_baseline, channel_number)
        timevalues_1 = hof_extract_brainstate_REM_wake(brain_state_1, brainstate)
        timevalues_2 = hof_extract_brainstate_REM_wake(brain_state_2, brainstate)
        withoutartifacts_1 = hof_filter(data_1, timevalues_1)
        withoutartifacts_1 = np.array(withoutartifacts_1)
        withoutartifacts_2 = hof_filter(data_2, timevalues_2)
        withoutartifacts_2 = np.array(withoutartifacts_2)
        intarray_1 = withoutartifacts_1.astype(int)
        intarray_2 = withoutartifacts_2.astype(int)
        higuchi_list_1 = []
        higuchi_list_2 = []
        for kmax_value in kmax_range:
            results_1 = [ant.higuchi_fd(epoch, kmax=kmax_value) for epoch in intarray_1]
            results_2 = [ant.higuchi_fd(epoch, kmax=kmax_value) for epoch in intarray_1]
            mean_hfd_1 = [mean(results_1)]
            mean_hfd_2 = [mean(results_2)]
            dataframe_1 = pd.DataFrame(data = {'Animal': animal, 'Channel': channel_number, 'Brainstate': brainstate, 'Kmax': kmax_value, 'HGF': mean_hfd_1})
            dataframe_2 = pd.DataFrame(data = {'Animal': animal, 'Channel': channel_number, 'Brainstate': brainstate, 'Kmax': kmax_value, 'HGF': mean_hfd_2})
            higuchi_df.append(dataframe_1)
            higuchi_df.append(dataframe_2)

merged_hfd = pd.concat(higuchi_df, axis=0).drop_duplicates().reset_index(drop=True)



mean_kmax_values = []

for kmax in kmax_range:
    kmax_average = merged_hfd[(merged_hfd['Kmax'] == kmax)]
    mean_kmax = [kmax_average['HGF'].mean()]
    mean_kmax_values.append(pd.DataFrame(data={'Kmax': kmax, 'HGF': mean_kmax}))

mean_df = pd.concat(mean_kmax_values, axis=0).drop_duplicates().reset_index(drop=True)
