
import math
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import re
import csv
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from datetime import timedelta
from pathlib import Path

# Initial Settings
plt.rcParams["figure.dpi"] = 100
plt.rcParams['font.size'] = 18
plt.rcParams["figure.figsize"] = 20, 4

buffer_seconds = 0
# output_foldername will be set dynamically via function argument
# graph_filename="raw" # This gets set in function
# mode="before_peakanalyzer" # This gets set in function

RRI_plot = False
autorange = False
ymin = 0
ymax = 4500
rri_autorange = False
rri_ymin = 0
rri_ymax = 10
segment_duration_seconds = 20

def create_dir(output_dir):
    dir = Path(output_dir)
    dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def from_timestamp_to_datetime(timestamp):
    jst = pytz.timezone('Asia/Tokyo')
    try:
        # Use timezone-aware datetime object directly from timestamp
        # The original code used utcfromtimestamp + 9 hours which is deprecated and can be improper for DST (though Japan doesn't have DST)
        # Using fromtimestamp with timezone is the modern way.
        dt = datetime.fromtimestamp(timestamp / 1000, jst)
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S') + f".{int(timestamp % 1000):03d}"
        return formatted_time
    except Exception:
         # Fallback to original logic if any issue, but fixing the deprecation warning
         utc_time = datetime.utcfromtimestamp(timestamp / 1000)
         jst_time = utc_time + timedelta(hours=9)
         formatted_time = jst_time.strftime('%Y-%m-%d %H:%M:%S')+f".{int(timestamp % 1000):03d}"
         return formatted_time

def from_datetime_to_timestamp(dt_str):
    jst = pytz.timezone("Asia/Tokyo")
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        dt = jst.localize(dt)
        timestamp = int(dt.timestamp()*1000)
        return timestamp
    except ValueError:
        # Handle cases where microsecond might be missing or different format
        return 0

def generate_graph(raw_df, filename, output_dir, mode, graph_filename_suffix):
    if mode == "before_peakanalyzer":
        raw_timestamp = raw_df.iloc[:, 0].to_numpy()
        raw_data = raw_df.iloc[:, 1].to_numpy()
    elif mode == "after_peakanalyzer":
        raw_timestamp = raw_df.iloc[:, 1].to_numpy()
        raw_data = raw_df.iloc[:, 2].to_numpy()
    
    graph_outputfolder = create_dir(os.path.join(output_dir, "graph"))
    
    raw_datetime = [from_timestamp_to_datetime(timestamp) for timestamp in raw_timestamp]
    raw_datetime = pd.to_datetime(raw_datetime)

    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.tick_params(axis='x', direction='in')
    ax1.tick_params(axis='y', direction='in', colors='black')
    ax1.grid(which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    ax1.plot(raw_datetime, raw_data, color='blue', linestyle='-', linewidth=2, alpha=0.6, label='raw_value')
    ax1.set_ylabel('ADC value', color="black")
    ax1.set_xlabel('datetime', color="black")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.3, -0.15), ncol=6)
    if autorange == False:
        ax1.set_ylim(ymin, ymax)
    
    plt.title(filename)
    fig.savefig(f"{graph_outputfolder}" + "/" + f"{filename}.jpg")
    fig.clf()
    plt.close("all")

def timestamp_separate(csv_rawdatafile, start, end, output_filenames, no, output_foldername, mode, graph_filename_suffix):
    # Base Output Dir
    base_output_dir = output_foldername
    output_dir_full_raw = create_dir(f"{base_output_dir}/full/raw")
    output_dir_20s_raw = create_dir(f"{base_output_dir}/20s_segments/raw")
    
    raw_df = pd.read_csv(csv_rawdatafile, header=0)
    
    raw_filename = os.path.basename(csv_rawdatafile)
    raw_filename_no_ext = os.path.splitext(raw_filename)[0]
    
    id_part = "unknown"
    if mode == "before_peakanalyzer":
        parts = raw_filename_no_ext.split('_')
        if len(parts) > 3:
            id_part = parts[3]
    elif mode == "after_peakanalyzer":
        parts = raw_filename_no_ext.split('_')
        if len(parts) > 6:
            id_part = parts[6]
        
    for i in range(len(start)):
        filtered_raw_df = raw_df[(raw_df['timestamp'] >= start[i]) & (raw_df['timestamp'] <= end[i])]
   
        jst = pytz.timezone('Asia/Tokyo')
        try:
            output_filename_obj = datetime.strptime(output_filenames[i], "%Y-%m-%d %H:%M:%S")
            output_filename_obj = jst.localize(output_filename_obj)
            starttime_obj = output_filename_obj.strftime("%Y%m%d_%H%M%S")
        except ValueError:
            starttime_obj = f"range_{i}"

        # Full file write
        raw_filename_full = f"clipdata_{no[i]}_{starttime_obj}_{id_part}_{graph_filename_suffix}.csv"
        
        raw_filepath_full = os.path.join(output_dir_full_raw, raw_filename_full)
        
        filtered_raw_df.to_csv(raw_filepath_full, index=False)
        
        graph_filename_full = os.path.splitext(raw_filename_full)[0]
        generate_graph(filtered_raw_df, graph_filename_full, output_dir_full_raw, mode, graph_filename_suffix)
        
        # 20s Segments
        clip_duration_ms = end[i] - start[i]
        clip_duration_s = clip_duration_ms / 1000.0
        num_segments = int(np.ceil(clip_duration_s / segment_duration_seconds))
        
        for seg_idx in range(num_segments):
            seg_start_ms = start[i] + (seg_idx * segment_duration_seconds * 1000)
            seg_end_ms = min(start[i] + ((seg_idx + 1) * segment_duration_seconds * 1000), end[i])
            
            seg_raw_df = filtered_raw_df[(filtered_raw_df['timestamp'] >= seg_start_ms) & (filtered_raw_df['timestamp'] <= seg_end_ms)]
            
            seg_suffix = f"_{seg_idx + 1}_"
            raw_filename_20s = f"clipdata_{no[i]}{seg_suffix}{starttime_obj}_{id_part}_{graph_filename_suffix}.csv"
            
            raw_filepath_20s = os.path.join(output_dir_20s_raw, raw_filename_20s)
            
            seg_raw_df.to_csv(raw_filepath_20s, index=False)
            
            graph_filename_20s = os.path.splitext(raw_filename_20s)[0]
            generate_graph(seg_raw_df, graph_filename_20s, output_dir_20s_raw, mode, graph_filename_suffix)

def read_ranges_from_csv(df):
    s_datetime = df.iloc[:, 0].to_numpy()
    e_datetime = df.iloc[:, 1].to_numpy()

    s_timestamp = [from_datetime_to_timestamp(dt) for dt in s_datetime] 
    e_timestamp = [from_datetime_to_timestamp(dt) for dt in e_datetime] 

    s_timestamp = np.array(s_timestamp)
    e_timestamp = np.array(e_timestamp)

    start_buffer = s_timestamp - buffer_seconds*1000
    end_buffer = e_timestamp + buffer_seconds*1000

    output_filenames = s_datetime
    return start_buffer, end_buffer, output_filenames

def main(csv_file, csv_rawdatafile, output_foldername_arg, graph_filename_suffix="raw", mode="before_peakanalyzer"):
    time_df = pd.read_csv(csv_file, header=0, dtype=str)
    start, end, output_filenames = read_ranges_from_csv(time_df)
    
    no = time_df.iloc[:, 2]
    no = no.str.replace('-', '_')
    
    print(f"Processing Job 1/X: {csv_file}")
    print(f" Output Folder: {output_foldername_arg}")
    
    timestamp_separate(csv_rawdatafile, start, end, output_filenames, no, output_foldername_arg, mode, graph_filename_suffix)

if __name__ == "__main__":
    # For testing, you could call main with hardcoded values if run directly
    pass
