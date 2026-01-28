
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

# --- Settings ---
plt.rcParams["figure.dpi"] = 100
plt.rcParams['font.size'] = 18
plt.rcParams["figure.figsize"] = 20, 4

buffer_seconds = 0
peak_mode = True
RRI_plot = False
autorange = True
ymin = -500
ymax = 500
rri_autorange = False
rri_ymin = 0
rri_ymax = 10
segment_duration_seconds = 20

def create_dir(output_dir):
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)

def from_timestamp_to_datetime(timestamp):
    try:
        jst_time = datetime.fromtimestamp(timestamp / 1000, pytz.timezone('Asia/Tokyo'))
        return jst_time.strftime('%Y-%m-%d %H:%M:%S') + f".{int(timestamp % 1000):03d}"
    except Exception as e:
        print(f"Error in timestamp conversion: {e}")
        return None

def from_datetime_to_timestamp(dt_str):
    try:
        jst = pytz.timezone("Asia/Tokyo")
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        dt_localized = jst.localize(dt)
        return int(dt_localized.timestamp() * 1000)
    except Exception as e:
        print(f"Error in datetime conversion: {e}")
        return None

def generate_graph(df, peak_df, filename, output_dir, peak_r_df=None):
    if df.empty:
        return

    graph_outputfolder = create_dir(os.path.join(output_dir, "graph"))

    timestamp = df.iloc[:, 1].to_numpy()
    data = df.iloc[:, 2].to_numpy()
    datetime_vals = pd.to_datetime([from_timestamp_to_datetime(ts) for ts in timestamp])

    peak_timestamp = peak_df.iloc[:, 1].to_numpy()
    peak_data = peak_df.iloc[:, 2].to_numpy()

    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(datetime_vals, data, color='blue', linestyle='-', linewidth=2, alpha=0.6, label='AD value')
    
    if peak_mode and not peak_df.empty:
        peak_datetime = pd.to_datetime([from_timestamp_to_datetime(ts) for ts in peak_timestamp])
        ax1.scatter(peak_datetime, peak_data, marker='o', edgecolors='blue', facecolors='none', label='Peak')

    if peak_mode and peak_r_df is not None and not peak_r_df.empty:
        peak_r_timestamp = peak_r_df.iloc[:, 1].to_numpy()
        peak_r_data = peak_r_df.iloc[:, 2].to_numpy()
        peak_r_datetime = pd.to_datetime([from_timestamp_to_datetime(ts) for ts in peak_r_timestamp])
        ax1.scatter(peak_r_datetime, peak_r_data, marker='o', edgecolors='red', facecolors='none', label='Reverse Peak')

    ax1.set_ylabel('ADC value', color="black")
    ax1.set_xlabel('datetime', color="black")
    if not autorange:
        ax1.set_ylim(ymin, ymax)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax1.grid(which='both', linestyle='--', linewidth=0.3, alpha=0.7)
    plt.title(f"{os.path.splitext(filename)[0]}_threshold[-10_4800]")

    save_path = os.path.join(graph_outputfolder, f"{os.path.splitext(filename)[0]}.jpg")
    fig.savefig(save_path)
    plt.close(fig)

def timestamp_separate(csv_datafile, start_timestamps, end_timestamps, output_filenames, no_list, output_foldername, graph_filename="HPF"):
    print(f"Processing: {csv_datafile}")
    
    base_output_dir = output_foldername
    base_filename_check = os.path.basename(csv_datafile)
    sub_folder_name = "data"
    if base_filename_check.startswith("peak_reverse"):
        sub_folder_name = "peak_reverse"
        
    output_dir_full_data = create_dir(f"{base_output_dir}/full/{sub_folder_name}")
    output_dir_20s_data = create_dir(f"{base_output_dir}/20s_segments/{sub_folder_name}")

    try:
        df = pd.read_csv(csv_datafile, header=0)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    base_filename = os.path.splitext(os.path.basename(csv_datafile))[0]
    filename_parts = base_filename.split('_')
    id_part = filename_parts[-1] if filename_parts else "unknown"
    
    peak_r_df = pd.DataFrame()

    if sub_folder_name == "peak_reverse":
        # peak_reverse file: time, timestamp, peak_value, dif_time, FWHM (5 cols)
        data_df = pd.DataFrame({
            "time": df.iloc[:,0].to_numpy(),
            "timestamp": df.iloc[:,1].to_numpy(),
            "value": df.iloc[:,2].to_numpy(),
        })
        peak_df = pd.DataFrame({
            "peak_time": df.iloc[:,0].to_numpy(),
            "peak_timestamp": df.iloc[:,1].to_numpy(),
            "peak_value": df.iloc[:,2].to_numpy(),
            "dif_time": df.iloc[:,3].to_numpy(),
        })
    else:
        # Standard: time, timestamp, value, peak_time, peak_timestamp, peak_value, dif_time (7+ cols)
        data_df = pd.DataFrame({
            "time": df.iloc[:,0].to_numpy(),
            "timestamp": df.iloc[:,1].to_numpy(),
            "value": df.iloc[:,2].to_numpy(),
        })
        peak_df = pd.DataFrame({
            "peak_time": df.iloc[:,3].to_numpy(),
            "peak_timestamp": df.iloc[:,4].to_numpy(),
            "peak_value": df.iloc[:,5].to_numpy(),
            "dif_time": df.iloc[:,6].to_numpy(),
        })
        
        # Load peak_reverse file if available
        base_name_only = os.path.basename(csv_datafile)
        dir_name = os.path.dirname(csv_datafile)
        if base_name_only.startswith("data_peak_pulse_"):
            original_base = base_name_only.replace("data_peak_pulse_", "")
            peak_r_filename = f"peak_reverse_{original_base}"
            peak_r_path = os.path.join(dir_name, peak_r_filename)
            if os.path.exists(peak_r_path):
                 try:
                     df_r = pd.read_csv(peak_r_path)
                     peak_r_df = pd.DataFrame({
                        "peak_time": df_r.iloc[:,0].to_numpy(),
                        "peak_timestamp": df_r.iloc[:,1].to_numpy(),
                        "peak_value": df_r.iloc[:,2].to_numpy(),
                     })
                 except Exception as e:
                     print(f"Warning: Could not read peak_reverse file: {e}")
        
        # Load Threshold Log file if available
        threshold_log_filename = f"threshold_log_{original_base}"
        threshold_log_path = os.path.join(dir_name, threshold_log_filename)
        threshold_df = pd.DataFrame()
        if os.path.exists(threshold_log_path):
             try:
                 threshold_df = pd.read_csv(threshold_log_path)
             except Exception as e:
                 print(f"Warning: Could not read threshold log file: {e}")
            
    # Output dir for sigma
    output_dir_sigma = create_dir(f"{base_output_dir}/full/sigma")
    output_dir_20s_sigma = create_dir(f"{base_output_dir}/20s_segments/sigma")

    for i in range(len(start_timestamps)):
        start_ts, end_ts = start_timestamps[i], end_timestamps[i]
        
        filtered_df = data_df[(data_df['timestamp'] >= start_ts) & (data_df['timestamp'] <= end_ts)]
        filtered_peak_df = peak_df[(peak_df['peak_timestamp'] >= start_ts) & (peak_df['peak_timestamp'] <= end_ts)]
        
        filtered_peak_r_df = pd.DataFrame()
        if not peak_r_df.empty:
            filtered_peak_r_df = peak_r_df[(peak_r_df['peak_timestamp'] >= start_ts) & (peak_r_df['peak_timestamp'] <= end_ts)]

        if filtered_df.empty and filtered_peak_df.empty:
            continue

        try:
            dt_obj = datetime.strptime(output_filenames[i], "%Y-%m-%d %H:%M:%S")
            starttime_str = dt_obj.strftime("%Y%m%d_%H%M%S")
        except ValueError:
            starttime_str = f"range_{i+1}"

        # Align peak data to start from beginning by resetting index
        if sub_folder_name == "peak_reverse":
            output_df = filtered_peak_df.reset_index(drop=True)
        else:
            output_df = pd.concat([filtered_df.reset_index(drop=True), filtered_peak_df.reset_index(drop=True)], axis=1)
        
        data_out_filename_full = f"clipdata_{no_list[i]}_{starttime_str}_{id_part}_{graph_filename}.csv"

        data_filepath_full = os.path.join(output_dir_full_data, data_out_filename_full)

        output_df.to_csv(data_filepath_full, index=False)

        generate_graph(filtered_df, filtered_peak_df, data_out_filename_full, output_dir_full_data, peak_r_df=filtered_peak_r_df)
        
        clip_duration_ms = end_ts - start_ts
        clip_duration_s = clip_duration_ms / 1000.0
        num_segments = int(np.ceil(clip_duration_s / segment_duration_seconds))
        
        for seg_idx in range(num_segments):
            seg_start_ms = start_ts + (seg_idx * segment_duration_seconds * 1000)
            seg_end_ms = min(start_ts + ((seg_idx + 1) * segment_duration_seconds * 1000), end_ts)
            
            seg_data_df = filtered_df[(filtered_df['timestamp'] >= seg_start_ms) & (filtered_df['timestamp'] <= seg_end_ms)]
            seg_peak_df = filtered_peak_df[(filtered_peak_df['peak_timestamp'] >= seg_start_ms) & (filtered_peak_df['peak_timestamp'] <= seg_end_ms)]
            
            seg_peak_r_df = pd.DataFrame()
            if not filtered_peak_r_df.empty:
                 seg_peak_r_df = filtered_peak_r_df[(filtered_peak_r_df['peak_timestamp'] >= seg_start_ms) & (filtered_peak_r_df['peak_timestamp'] <= seg_end_ms)]
            
            seg_suffix = f"_{seg_idx + 1}_"
            data_out_filename_20s = f"clipdata_{no_list[i]}{seg_suffix}{starttime_str}_{id_part}_{graph_filename}.csv"
            
            data_filepath_20s = os.path.join(output_dir_20s_data, data_out_filename_20s)
            
            # Align peak data to start from beginning
            seg_output_df = pd.concat([seg_data_df.reset_index(drop=True), seg_peak_df.reset_index(drop=True)], axis=1)
            seg_output_df.to_csv(data_filepath_20s, index=False)
            
            generate_graph(seg_data_df, seg_peak_df, data_out_filename_20s, output_dir_20s_data, peak_r_df=seg_peak_r_df)

            # --- Filter and Save Threshold Log (20s segment) ---
            if sub_folder_name != "peak_reverse" and not threshold_df.empty:
                seg_filtered_threshold_df = threshold_df[
                    (threshold_df['start_timestamp'] <= seg_end_ms) & 
                    (threshold_df['end_timestamp'] >= seg_start_ms)
                ]
                
                if not seg_filtered_threshold_df.empty:
                     sigma_out_filename_20s = f"clipdata_{no_list[i]}{seg_suffix}{starttime_str}_{id_part}_threshold.csv"
                     sigma_filepath_20s = os.path.join(output_dir_20s_sigma, sigma_out_filename_20s)
                     seg_filtered_threshold_df.to_csv(sigma_filepath_20s, index=False)

        # --- Filter and Save Threshold Log (Full Clip) ---
        if sub_folder_name != "peak_reverse" and not threshold_df.empty:
            # Overlap condition: (Target.Start <= Clip.End) and (Target.End >= Clip.Start)
            # threshold_df has 'start_timestamp' and 'end_timestamp'
            filtered_threshold_df = threshold_df[
                (threshold_df['start_timestamp'] <= end_ts) & 
                (threshold_df['end_timestamp'] >= start_ts)
            ]
            
            if not filtered_threshold_df.empty:
                 sigma_out_filename = f"clipdata_{no_list[i]}_{starttime_str}_{id_part}_threshold.csv"
                 sigma_filepath = os.path.join(output_dir_sigma, sigma_out_filename)
                 filtered_threshold_df.to_csv(sigma_filepath, index=False)

def read_ranges_from_csv(df):
    try:
        s_datetime_str = df.iloc[:, 0].to_numpy(dtype=str)
        e_datetime_str = df.iloc[:, 1].to_numpy(dtype=str)

        s_timestamp = np.array([from_datetime_to_timestamp(dt) for dt in s_datetime_str])
        e_timestamp = np.array([from_datetime_to_timestamp(dt) for dt in e_datetime_str])

        start_buffer = s_timestamp - buffer_seconds * 1000
        end_buffer = e_timestamp + buffer_seconds * 1000

        return start_buffer, end_buffer, s_datetime_str
    except Exception as e:
        print(f"Error reading ranges: {e}")
        return None, None, None

def main(csv_file, csv_datafile, output_foldername_arg, graph_filename="HPF"):
    # csv_file: Time Range
    # csv_datafile: Data (Formatted/Analyzed)
    
    time_df = pd.read_csv(csv_file, header=0, dtype=str)
    start_timestamps, end_timestamps, output_filenames = read_ranges_from_csv(time_df)
    
    no_list = time_df.iloc[:, 2]
    no_list = no_list.str.replace('-', '_')
    
    if start_timestamps is not None:
        timestamp_separate(csv_datafile, start_timestamps, end_timestamps, output_filenames, no_list, output_foldername_arg, graph_filename)

if __name__ == "__main__":
    pass
