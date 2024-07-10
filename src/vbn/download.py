import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="data/vbn_data/")
parser.add_argument('--bin_step_size', type=int, default=0.001, help="The size of the step in each bin if set to 0.001=1ms so in the experiments each interval will be 50ms interval")  
parser.add_argument('--cache_dir', type=str, default="data/vbn_cache/")  



from tqdm import tqdm

def makePSTH(spikes, startTimes, windowDur, binSize=0.01):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/startTimes.size
    return counts/binSize, bins


def makebins(spikes, startTime, nb_bins = 5 ,windowDur = 50, unit=0.001):
    bins = np.zeros((nb_bins))
    bin_size_ms = windowDur * unit
    np.sort(spikes)
    for bin in range(nb_bins):
        startInd = np.searchsorted(spikes, startTime+ bin * bin_size_ms )
        endInd = np.searchsorted(spikes, startTime+ (bin+1) * bin_size_ms)
        
      
        bins[bin] = endInd - startInd
    
    return bins

def is_licking(times, timestep_start,timestep_end):
    if hasattr(times, "__len__")==False:
        times = np.array([times])
    else:
        times = np.array(times)
    if times.shape[0] ==0:
        return False
    else:
        for t in times:
            if t >= timestep_start and t<= timestep_end:
                return True
         
        return False
                

def add_nb_flash(df): 
    out = df.copy()
    out["flash_nb"] = np.zeros(out.shape[0])
    out["is_licking"] = np.zeros(out.shape[0])
    out["prev_omitted"] = np.zeros(out.shape[0])
    ou_d = df.reset_index()
    for idx, row in out.iterrows():
        q = ou_d.query('trials_id == {}'.format(row["trials_id"]))
        q.sort_values(by=['start_time_x'], ascending=True)
        i=0
        omitted= False
        for idx, row_x in q.iterrows():
            i+=1
            st_id =row_x["stimulus_presentations_id"]
            out.loc[st_id ,"flash_nb"] = i
            out.loc[st_id ,"is_licking"] = is_licking(row_x["lick_times"], row_x["start_time_x"],row_x["end_time"])
            out.loc[st_id ,"prev_omitted"] = omitted
            omitted = row_x["omitted"]
            
    #out["is_licking"] =  out[["start_time_x","lick_times"]].apply(lambda x : False if len(x)== None or len(x)==0  else True)
       
    out["rewarded"] =  out["reward_time"].apply(lambda x : False if x== None  else True) 
    return out

def session_id_process(session_id,bin_step_size = 0.01,areas_good= None):
        
        session = cache.get_ecephys_session(
            ecephys_session_id=session_id)
    
        units = session.get_units()
        channels = session.get_channels()
        spike_times = session.spike_times
        unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
        count_unist = unit_channels.value_counts('structure_acronym')
        ## get areas_acronyms > 20 units
        # areas_good = count_unist.where(count_unist>20).dropna().index
        
        
        # print(areas_good)
        # areas_good = areas_good[:2]
        ## cross with trials ans stimulus to get all info
        stimulus_presentations = session.stimulus_presentations[ session.stimulus_presentations["stimulus_block"]==0]
        stim_trials_table = stimulus_presentations.merge(session.trials,left_on= "trials_id", right_index=True)
        
        
        full_tab = add_nb_flash(stim_trials_table)
        
        
        flashes_for_no_change = np.r_[4:11] 
        non_change_times = full_tab[full_tab['active'] & 
                       full_tab['flash_nb'].isin(flashes_for_no_change)&
                       ~full_tab['is_change_x'] & 
                       full_tab['rewarded'] & 
                       (full_tab['image_name'] == full_tab['initial_image_name'])&
                       ~full_tab["is_licking"] & 
                       ~full_tab["omitted"]]['start_time_x'].values

        change_times = full_tab[full_tab['active'] & full_tab['is_change_x'] & full_tab['rewarded'] ]['start_time_x'].values
        
        
        #now we'll filter them
        good_unit_filter = ((unit_channels['snr']>1)&
                            (unit_channels['isi_violations']<1)
                           # &(unit_channels['firing_rate']>0.1)
                            )
        good_units = unit_channels.loc[good_unit_filter]
        
        
        data_area = {}
        for ar in areas_good:
            out_change, bins = get_time_series(change_times,ar,good_units,spike_times,step_bin=bin_step_size)
            
            out_non_change, bins = get_time_series(non_change_times,ar,good_units,spike_times,step_bin=bin_step_size)
            
            
            data_area[ar] ={
                "change": out_change,
                "non_change" :out_non_change
            }
        return data_area, non_change_times, change_times , 

def get_time_series(times,area,good_units,spike_times,step_bin):
    area_of_interest = area
    area_units = good_units[good_units['structure_acronym']==area_of_interest]
    time_before_change = 0
    duration = 2.5
    flashes = []
    bins = None
    for time in times:
        unit_flash =[]
        for iu, unit in area_units.iterrows():
            unit_spike_times = spike_times[iu]
            unit_change_response = makebins(unit_spike_times, time)
            
            # makePSTH(spikes = unit_spike_times, 
            #                                 startTimes = np.array([time-time_before_change]), 
            #                                 windowDur = duration, 
            #                                 binSize=step_bin)
            
            unit_flash.append(unit_change_response)
        flashes.append(unit_flash)

    flashes = np.array(flashes)
    return flashes,bins



def save_session(session_id,dict,path):
    file_path = os.path.join(path,str(session_id)) 
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    with  open(os.path.join(file_path ,'all_reg.pickle' ) , 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_session(session_id,path):
    file_path = os.path.join(path,str(session_id)) 
    file_path = os.path.join(file_path,'all_reg.pickle') 
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
    return b 

if __name__=="__main__":
    
    
    args = parser.parse_args()
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=args.cache_dir)

    # get the metadata tables
    units_table = cache.get_unit_table()

    channels_table = cache.get_channel_table()

    probes_table = cache.get_probe_table()

    behavior_sessions_table = cache.get_behavior_session_table()

    ecephys_sessions_table = cache.get_ecephys_session_table()

    ecephys_sessions_table_idx = ecephys_sessions_table.reset_index()
    session_ids = ecephys_sessions_table_idx['ecephys_session_id'].values

    num_units_df = (
            units_table
            .groupby(['ecephys_session_id', 'structure_acronym'])
            ['structure_acronym']
            .count()
        )

    num_units_df = (
            num_units_df
            .unstack()
            .fillna(0)
            .astype(int)
        )
    #structures = ['VISp', 'VISl', 'VISal']
    structures = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm" ]
    num_units_df = num_units_df.loc[session_ids]

    min_unit_count_thresh = 20
    session_indices = num_units_df.index[
            (num_units_df[structures] > min_unit_count_thresh).all(axis=1)
        ]

    mice_with_both_fam_and_nov = (
            ecephys_sessions_table_idx
            .set_index('ecephys_session_id')
            .loc[session_indices]
            .reset_index()
            .set_index(['mouse_id', 'experience_level'])
            .sort_index()
            ['ecephys_session_id']
            .unstack()
            .dropna()
            .astype('int')
            .stack()
        )
    
    ret={}
    all_sessions =list( ecephys_sessions_table.index ) 
    good_sess = []
    
    for i, session_id in enumerate(mice_with_both_fam_and_nov):
        good_sess.append(session_id)
    
    with open("data/good_sessions.pickle", "wb") as f:
        pickle.dump(good_sess, f)
    
    
    for i, session_id in tqdm( enumerate(good_sess[:1]) ):
        print(session_id)
        data, non_change, change = session_id_process(session_id, bin_step_size = args.bin_step_size,areas_good=structures )
        save_session(session_id,data,"data/vbn_{}/".format(args.bin_step_size))
        