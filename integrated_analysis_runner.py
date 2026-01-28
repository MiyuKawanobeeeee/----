
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import shutil
import pandas as pd
from datetime import datetime

# Import libraries
# Assuming they are in the same folder. If not, add system path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lib_job1
import lib_job2
import lib_job5
import lib_job7
import lib_job8
import glob


class Task7ConfigWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Task 7 Parameter Input (diftime)")
        self.geometry("800x600")
        self.transient(master)
        
        self.method_vars = {}
        self.mode_widgets = {}
        self.mode_frames = {}
        
        # Widgets by Page
        self.p1_widgets = {}
        self.p2_widgets = {}
        self.p3_widgets = {} # For common items in P3 tab like ID/Excl
        
        self.create_widgets()
        
    def create_widgets(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab Control
        tab_control = ttk.Notebook(container)
        tab_control.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: Peak 2 (Page 1)
        tab_p1 = ttk.Frame(tab_control)
        tab_control.add(tab_p1, text="Peak 2 (Page 1)")
        self.create_p1_inputs(tab_p1)

        # Tab 2: Mountain (Page 2)
        tab_p2 = ttk.Frame(tab_control)
        tab_control.add(tab_p2, text="Mountain (Page 2)")
        self.create_p2_inputs(tab_p2)

        # Tab 3: Time Ratio (Page 3)
        tab_p3 = ttk.Frame(tab_control)
        tab_control.add(tab_p3, text="Time Ratio (Page 3)")
        self.create_p3_inputs(tab_p3)
        
        # Save Button
        save_btn = ttk.Button(container, text="Save All Parameters", command=self.on_save)
        save_btn.pack(pady=10)
        
    def create_id_input(self, parent, widget_dict):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="ID Extraction (Start_End):").pack(side=tk.LEFT)
        e_s = ttk.Entry(f, width=5)
        e_s.insert(0, "2")
        e_s.pack(side=tk.LEFT, padx=5)
        e_e = ttk.Entry(f, width=5)
        e_e.insert(0, "4")
        e_e.pack(side=tk.LEFT, padx=5)
        
        widget_dict['split_start'] = e_s
        widget_dict['split_end'] = e_e

    def create_p1_inputs(self, parent):
        # ID Settings
        self.create_id_input(parent, self.p1_widgets)
        
        defaults = {"interval": "2.0", "ratio_val": "70", "target_peak": "2"}
        
        frame = ttk.Frame(parent)
        frame.pack(padx=10, pady=10, fill=tk.X)
        
        ttk.Label(frame, text="Interval Width (Hz):").grid(row=0, column=0, sticky="w", pady=5)
        e_int = ttk.Entry(frame, width=10)
        e_int.insert(0, defaults["interval"])
        e_int.grid(row=0, column=1, padx=5)
        self.p1_widgets["interval"] = e_int
        
        ttk.Label(frame, text="Ratio Threshold (%):").grid(row=1, column=0, sticky="w", pady=5)
        e_rat = ttk.Entry(frame, width=10)
        e_rat.insert(0, defaults["ratio_val"])
        e_rat.grid(row=1, column=1, padx=5)
        self.p1_widgets["ratio_val"] = e_rat
        
        ttk.Label(frame, text="Max Target Peak Index (2 or 3):").grid(row=2, column=0, sticky="w", pady=5)
        e_mtp = ttk.Entry(frame, width=10)
        e_mtp.insert(0, defaults["target_peak"])
        e_mtp.grid(row=2, column=1, padx=5)
        self.p1_widgets["target_peak"] = e_mtp

        # Removed Y Max (Auto)

    def create_p2_inputs(self, parent):
        # ID Settings
        self.create_id_input(parent, self.p2_widgets)
        
        defaults = {"int_peak": "2.0", "int_sum": "1.0"}
        
        frame = ttk.Frame(parent)
        frame.pack(padx=10, pady=10, fill=tk.X)
        
        ttk.Label(frame, text="Interval Width (Peak Conv) (Hz):").grid(row=0, column=0, sticky="w", pady=5)
        e_peak = ttk.Entry(frame, width=10)
        e_peak.insert(0, defaults["int_peak"])
        e_peak.grid(row=0, column=1, padx=5)
        self.p2_widgets["int_peak"] = e_peak
        
        ttk.Label(frame, text="Interval Width (Sum Conv) (Hz):").grid(row=1, column=0, sticky="w", pady=5)
        e_sum = ttk.Entry(frame, width=10)
        e_sum.insert(0, defaults["int_sum"])
        e_sum.grid(row=1, column=1, padx=5)
        self.p2_widgets["int_sum"] = e_sum
        
        # Removed Y Max (Auto)

    def create_p3_inputs(self, parent):
        # ID Settings
        self.create_id_input(parent, self.p3_widgets)

        # Threshold
        th_frame = ttk.Frame(parent)
        th_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(th_frame, text="Exclusion Threshold (Point):").pack(side=tk.LEFT)
        self.exc_th = ttk.Entry(th_frame, width=5)
        self.exc_th.insert(0, "20")
        self.exc_th.pack(side=tk.LEFT, padx=5)
        self.p3_widgets["exc_th"] = self.exc_th

        # Mode Selection
        mode_frame = ttk.LabelFrame(parent, text="Analysis Modes (Select at least one)")
        mode_frame.pack(fill=tk.X, pady=5, padx=5)
        
        modes = [
            "指定時間で１判定",
            "指定時間で１判定(平均化)",
            "指定時間で１判定(最頻値)",
            "指定時間で１判定(自動)",
            "指定時間で１判定(平均値基準)",
            "全体最頻値基準", 
            "1ファイルにつき1回判定"
        ]
        
        for m in modes:
            var = tk.BooleanVar(value=True if m == "1ファイルにつき1回判定" else False)
            self.method_vars[m] = var
            cb = ttk.Checkbutton(mode_frame, text=m, variable=var)
            cb.pack(anchor=tk.W, padx=10)

        # Parameter Detail
        param_frame = ttk.LabelFrame(parent, text="Parameter Details (Select Mode to Edit)")
        param_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        self.mode_combobox = ttk.Combobox(param_frame, values=modes, state="readonly")
        self.mode_combobox.pack(pady=5)
        self.mode_combobox.bind("<<ComboboxSelected>>", self.on_mode_select)
        self.mode_combobox.current(0)
        
        self.param_container = ttk.Frame(param_frame)
        self.param_container.pack(fill=tk.BOTH, expand=True, padx=5)
        
        for m in modes:
            f = ttk.Frame(self.param_container)
            self.mode_frames[m] = f
            self.create_mode_inputs_p3(f, m)
            
        self.on_mode_select(None)
        
    def create_mode_inputs_p3(self, parent, mode_name):
        widgets = {}
        defaults = {
            "group_time": "2", "top_n": "13",
            "ratio1": "8/1", "th1": "0.4",
            "ratio2": "13/1", "th2": "0.2",
            "offset": "3", "center": "0"
        }
        
        # Row 0
        tk.Label(parent, text="Group Time (s):").grid(row=0, column=0, sticky="w")
        e_time = tk.Entry(parent, width=10)
        e_time.insert(0, defaults["group_time"])
        e_time.grid(row=0, column=1, padx=5, pady=2)
        widgets["group_time"] = e_time
        
        tk.Label(parent, text="Top N:").grid(row=0, column=2, sticky="w")
        e_top = tk.Entry(parent, width=10)
        e_top.insert(0, defaults["top_n"])
        e_top.grid(row=0, column=3, padx=5, pady=2)
        widgets["top_n"] = e_top
        
        # Row 1
        tk.Label(parent, text="Ratio 1:").grid(row=1, column=0, sticky="w")
        e_r1 = tk.Entry(parent, width=10)
        e_r1.insert(0, defaults["ratio1"])
        e_r1.grid(row=1, column=1, padx=5, pady=2)
        widgets["ratio1"] = e_r1
        
        tk.Label(parent, text="Threshold 1:").grid(row=1, column=2, sticky="w")
        e_th1 = tk.Entry(parent, width=10)
        e_th1.insert(0, defaults["th1"])
        e_th1.grid(row=1, column=3, padx=5, pady=2)
        widgets["th1"] = e_th1
        
        # Row 2
        tk.Label(parent, text="Ratio 2:").grid(row=2, column=0, sticky="w")
        e_r2 = tk.Entry(parent, width=10)
        e_r2.insert(0, defaults["ratio2"])
        e_r2.grid(row=2, column=1, padx=5, pady=2)
        widgets["ratio2"] = e_r2
        
        tk.Label(parent, text="Threshold 2:").grid(row=2, column=2, sticky="w")
        e_th2 = tk.Entry(parent, width=10)
        e_th2.insert(0, defaults["th2"])
        e_th2.grid(row=2, column=3, padx=5, pady=2)
        widgets["th2"] = e_th2
        
        if "平均値基準" in mode_name or "全体最頻値基準" in mode_name:
             tk.Label(parent, text="Offset:").grid(row=3, column=0, sticky="w")
             e_off = tk.Entry(parent, width=10)
             e_off.insert(0, defaults["offset"])
             e_off.grid(row=3, column=1, padx=5, pady=2)
             widgets["offset"] = e_off
             
             tk.Label(parent, text="Center (0=Auto):").grid(row=3, column=2, sticky="w")
             e_ctr = tk.Entry(parent, width=10)
             e_ctr.insert(0, defaults["center"])
             e_ctr.grid(row=3, column=3, padx=5, pady=2)
             widgets["center"] = e_ctr
             
        self.mode_widgets[mode_name] = widgets

    def on_mode_select(self, event):
        selected = self.mode_combobox.get()
        for m, f in self.mode_frames.items():
            f.pack_forget()
        if selected in self.mode_frames:
            self.mode_frames[selected].pack(fill=tk.BOTH, expand=True)

    def on_save(self):
        try:
            # P1 Data
            p1_config = {
                'interval': self.p1_widgets['interval'].get(),
                'ratio_val': self.p1_widgets['ratio_val'].get(),
                'target_peak': self.p1_widgets['target_peak'].get(),
                'split_start': self.p1_widgets['split_start'].get(),
                'split_end': self.p1_widgets['split_end'].get()
            }
            
            # P2 Data
            p2_config = {
                'int_peak': self.p2_widgets['int_peak'].get(),
                'int_sum': self.p2_widgets['int_sum'].get(),
                'split_start': self.p2_widgets['split_start'].get(),
                'split_end': self.p2_widgets['split_end'].get()
            }

            # P3 Data
            selected_modes = [m for m, v in self.method_vars.items() if v.get()]
            if not selected_modes:
                messagebox.showwarning("Warning", "Select at least one Page 3 mode.")
                return
            
            p3_configs = []
            exc_th = self.p3_widgets['exc_th'].get()  # From P3 specific widget dict
            p3_split_start = self.p3_widgets['split_start'].get()
            p3_split_end = self.p3_widgets['split_end'].get()
            
            for m in selected_modes:
                w = self.mode_widgets[m]
                cfg = {
                    'mode_name': m,
                    'group_time': w['group_time'].get(),
                    'top_n': w['top_n'].get(),
                    'ratio1': w['ratio1'].get(),
                    'ratio2': w['ratio2'].get(),
                    'th1': w['th1'].get(),
                    'th2': w['th2'].get(),
                    'split_start': p3_split_start,
                    'split_end': p3_split_end,
                    'exclusion_th': exc_th
                }
                if "offset" in w: cfg['offset'] = w['offset'].get()
                if "center" in w: cfg['center'] = w['center'].get()
                p3_configs.append(cfg)
                
            self.master.task7_params = {
                "p1": p1_config,
                "p2": p2_config,
                "p3": p3_configs
            }
            self.master.log(f"Parameters Saved. P3 Modes: {len(p3_configs)}")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")

class IntegratedRunnerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Integrated Automated Analysis Pipeline (diftime処理追加)")
        self.geometry("900x700")
        
        # Initialize log widget early so self.log(...) works immediately
        self.log_text = tk.Text(self, height=10)
        
        self.task7_params = None
        # Base directory is this script's folder
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log(f"Base Project Directory: {self.base_dir}")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_task1_tab()
        self.create_task2_tab()
        self.create_task3_tab() # Copy
        self.create_task4_tab() # Like Task 2
        self.create_task5_tab() # Copy
        self.create_task6_tab() # FFT
        self.create_task7_tab() # Final Analysis
        self.create_task8_tab() # New Task 8 (Kawanobe & Diftime)
        self.create_task9_tab() # Old Task 8 (Karte Data)
        
        # Pack the log widget at the bottom (created earlier)
        self.log_text.pack(fill=tk.X, padx=10, pady=5)
        
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        print(message)

    def open_task7_config(self):
        Task7ConfigWindow(self)

    def select_raw_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if paths:
            self.t1_raw_files = list(paths)
            self.t1_raw.delete(0, tk.END)
            if len(paths) == 1:
                self.t1_raw.insert(0, paths[0])
            else:
                self.t1_raw.insert(0, f"({len(paths)} files selected)")
        else:
            pass

    def select_file(self, entry_widget):
        path = filedialog.askopenfilename()
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def select_folder(self, entry_widget):
        path = filedialog.askdirectory()
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def create_task1_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 1")
        
        ttk.Label(tab, text="Input Time CSV (Processes all 'cliptime*.csv' in the selected file's folder):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t1_csv = ttk.Entry(tab, width=50)
        self.t1_csv.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_file(self.t1_csv)).grid(row=0, column=2)
        
        ttk.Label(tab, text="Raw Data CSV (Select one or multiple files):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.t1_raw = ttk.Entry(tab, width=50)
        self.t1_raw.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=self.select_raw_files).grid(row=1, column=2)
        
        
        ttk.Label(tab, text="Output Folder Name:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.t1_out = ttk.Entry(tab, width=50)
        self.t1_out.insert(0, os.path.join(self.base_dir, "clipdata_NK04_raw"))
        self.t1_out.grid(row=3, column=1, padx=5, pady=5)
        
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=4, column=1, pady=20)
        
        ttk.Button(btn_frame, text="Run Task 1", command=self.run_task1).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Run Task 1-7 (Batch)", command=self.run_batch_task1_to_7).pack(side=tk.LEFT, padx=5)

    def run_task1(self):
        self.log("Running Task 1...")
        input_path = self.t1_csv.get()
        raw_entry_val = self.t1_raw.get()
        out_folder = self.t1_out.get()
        
        if not input_path or not os.path.exists(input_path):
             self.log(f"Error: Input path not found: {input_path}")
             return False
        
        # Determine raw files
        raw_files = []
        if hasattr(self, 't1_raw_files') and self.t1_raw_files and "files selected" in raw_entry_val:
            raw_files = self.t1_raw_files
        elif raw_entry_val and os.path.exists(raw_entry_val):
            raw_files = [raw_entry_val]
        else:
            self.log("Error: No valid Raw Data CSV selected.")
            return False

        # Determine directory to search for cliptime files
        if os.path.isfile(input_path):
            search_dir = os.path.dirname(input_path)
        else:
            search_dir = input_path
            
        # Find all cliptime*.csv
        cliptime_files = glob.glob(os.path.join(search_dir, "cliptime*.csv"))
        if not cliptime_files:
             self.log(f"No 'cliptime*.csv' files found in {search_dir}")
             return False
        
        self.log(f"Found {len(cliptime_files)} cliptime files and {len(raw_files)} raw data files.")
        
        success_count = 0
        error_count = 0
        
        for r_file in raw_files:
            for t_file in cliptime_files:
                try:
                    self.log(f"  Processing Time:{os.path.basename(t_file)} vs Raw:{os.path.basename(r_file)}...")
                    lib_job1.main(t_file, r_file, out_folder)
                    success_count += 1
                except Exception as e:
                    self.log(f"  Error processing Pair ({os.path.basename(t_file)}, {os.path.basename(r_file)}): {e}")
                    error_count += 1
                
        self.log(f"Task 1 Completed. Success: {success_count}, Errors: {error_count}")
        return success_count > 0

    def create_task2_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 2")
        
        ttk.Label(tab, text="Input Time CSV (Processes all 'cliptime*.csv' in the selected file's folder):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t2_csv = ttk.Entry(tab, width=50)
        self.t2_csv.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_file(self.t2_csv)).grid(row=0, column=2)
        
        # Displaying target folders logic
        ttk.Label(tab, text="Processed Data (Auto-scans specific folder):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.t2_data = ttk.Entry(tab, width=50, state="disabled")
        self.t2_data.insert(0, "Auto-scans: ...[-10_4800]_HPF8.0Hz_脈")
        self.t2_data.grid(row=1, column=1, padx=5, pady=5)
        
        # Information Note
        note_frame = ttk.LabelFrame(tab, text="Target Folder for Processed Data")
        note_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        ttk.Label(note_frame, text="Target: HPF8Hz_0.55_0.30_0.85_autothre_ave1_3シグマ", justify=tk.LEFT).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(tab, text="Output Folder Name:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.t2_out = ttk.Entry(tab, width=50)
        self.t2_out.insert(0, os.path.join(self.base_dir, "clipdata_NK04_HPF"))
        self.t2_out.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Button(tab, text="Run Task 2", command=self.run_task2).grid(row=4, column=1, pady=20)

    def run_task2(self):
        self.log("Running Task 2...")
        input_path = self.t2_csv.get()
        if not input_path:
            input_path = self.t1_csv.get()
            self.log(f"Task 2 Input empty, using Task 1 input: {input_path}")
            
        out_folder = self.t2_out.get()
        
        # 1. Input Time CSVs (Same as Task 1 logic)
        if not input_path or not os.path.exists(input_path):
             self.log(f"Error: Input path not found: {input_path}")
             return False
             
        if os.path.isfile(input_path):
            search_dir = os.path.dirname(input_path)
        else:
            search_dir = input_path
            
        cliptime_files = glob.glob(os.path.join(search_dir, "cliptime*.csv"))
        if not cliptime_files:
             self.log(f"No 'cliptime*.csv' files found in {search_dir}")
             return False
             
        # 2. Processed Data CSVs (Specific Folder)
        target_folder_name = "HPF8Hz_0.55_0.30_0.85_autothre_ave1_3シグマ"
        folder_full_path = os.path.join(search_dir, target_folder_name)
        
        data_files = []
        if os.path.isdir(folder_full_path):
            # glob fails with brackets in path, so use os.listdir
            try:
                all_files = os.listdir(folder_full_path)
                found = [os.path.join(folder_full_path, f) for f in all_files if (f.startswith("data_peak_pulse_") or f.startswith("peak_reverse")) and f.endswith(".csv")]
                self.log(f"Found {len(found)} files in {target_folder_name}")
                data_files.extend(found)
            except Exception as e:
                self.log(f"Error reading directory {folder_full_path}: {e}")
        else:
            self.log(f"Warning: Folder not found: {folder_full_path}")
                
        if not data_files:
             self.log("No 'data_peak_pulse_*.csv' or 'peak_reverse*' files found in target folder.")
             return False
        
        # 3. Execution Loop
        success_count = 0
        error_count = 0
        
        for t_file in cliptime_files:
            for d_file in data_files:
                try:
                    lib_job2.main(t_file, d_file, out_folder)
                    success_count += 1
                except Exception as e:
                    self.log(f"  Error processing {os.path.basename(t_file)}: {e}")
                    error_count += 1
                    
        self.log(f"Task 2 Completed. Success: {success_count}, Errors: {error_count}")
        return success_count > 0

    def create_task3_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 3 (Copy)")
        
        ttk.Label(tab, text="Source Folder (Task 2 Output):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t3_src = ttk.Entry(tab, width=50)
        self.t3_src.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t3_src)).grid(row=0, column=2)
        
        
        ttk.Button(tab, text="Run Task 3", command=self.run_task3).grid(row=1, column=1, pady=20)

    def run_task3(self):
        self.log("Running Task 3...")
        src_root = self.t3_src.get()
        if not src_root:
            src_root = self.t2_out.get()
            self.log(f"Input empty, using Task 2 output: {src_root}")
            
        # Requirement:
        # 1. 20s_segments/data -> 第２ピーク (Peak 2)
        # 2. full/data -> 山 (Mountain)
        
        try:
            # 1. Copy 20s_segments/data to 第２ピーク
            src_20s = os.path.join(src_root, "20s_segments", "data")
            dest_peak2 = os.path.join(os.path.dirname(os.path.abspath(src_root)), "第２ピーク")
            
            if os.path.exists(src_20s):
                os.makedirs(dest_peak2, exist_ok=True)
                count_20s = 0
                for f in os.listdir(src_20s):
                    if f.endswith(".csv"):
                        shutil.copy2(os.path.join(src_20s, f), dest_peak2)
                        count_20s += 1
                self.log(f"Copied {count_20s} files from {src_20s} to {dest_peak2}")
            else:
                self.log(f"[Warning] Source folder not found: {src_20s}")

            # 2. Copy full/data to 山
            src_full = os.path.join(src_root, "full", "data")
            dest_mountain = os.path.join(os.path.dirname(os.path.abspath(src_root)), "山")
            
            if os.path.exists(src_full):
                os.makedirs(dest_mountain, exist_ok=True)
                count_full = 0
                for f in os.listdir(src_full):
                    if f.endswith(".csv"):
                        shutil.copy2(os.path.join(src_full, f), dest_mountain)
                        count_full += 1
                self.log(f"Copied {count_full} files from {src_full} to {dest_mountain}")
            else:
                self.log(f"[Warning] Source folder not found: {src_full}")

            self.log("Task 3 Completed.")
            return True

        except Exception as e:
            self.log(f"Error in Task 3: {e}")
            return False

    def run_batch_task1_to_7(self):
        self.log("=== Starting Batch Execution (Tasks 1-7) ===")
        
        # Task 1
        if not self.run_task1():
            self.log("Batch stopped due to Task 1 failure.")
            return

        # Task 2
        if not self.run_task2():
            self.log("Batch stopped due to Task 2 failure.")
            return

        # Task 3
        if not self.run_task3():
            self.log("Batch stopped due to Task 3 failure.")
            return
        
        # Infer Task 3 Output Base (Parent of 第２ピーク/山)
        # T3 uses T3 src or T2 out.
        t3_src_root = self.t3_src.get()
        if not t3_src_root: t3_src_root = self.t2_out.get()
        base_t3 = os.path.dirname(os.path.abspath(t3_src_root))
        
        p2_folder_path = os.path.join(base_t3, "第２ピーク")
        mt_folder_path = os.path.join(base_t3, "山")

        # Task 4
        if not self.run_task4():
             self.log("Batch stopped due to Task 4 failure.")
             return
            
        # Task 5
        if not self.run_task5():
             self.log("Batch stopped due to Task 5 failure.")
             return
             
        # Infer Task 5 Output
        t5_src_root = self.t5_src.get()
        if not t5_src_root: t5_src_root = self.t4_out.get()
        base_t5 = os.path.dirname(os.path.abspath(t5_src_root))
        tr_folder_path = os.path.join(base_t5, "時間区間比率")

        # Task 6
        # T6 uses T6 Src or T2 Out Parent (which is likely base_t3)
        # Let's ensure T6 uses the correct base from T3
        self.t6_src.delete(0, tk.END)
        self.t6_src.insert(0, base_t3)
        
        if not self.run_task6():
             self.log("Batch stopped due to Task 6 failure.")
             return
             
        # Task 7 Setup
        self.log("Setting up Task 7 Inputs automatically...")
        
        self.t7_p2_res.delete(0, tk.END)
        self.t7_p2_res.insert(0, p2_folder_path)
        
        self.t7_mt_res.delete(0, tk.END)
        self.t7_mt_res.insert(0, mt_folder_path)
        
        self.t7_tr.delete(0, tk.END)
        self.t7_tr.insert(0, tr_folder_path)
        
        # Task 7
        if not self.run_task7():
             self.log("Batch stopped due to Task 7 failure.")
             return

        # Task 8
        if not self.run_task8():
             self.log("Batch stopped due to Task 8 failure.")
             return

        # Task 9
        if not self.run_task9():
             self.log("Batch stopped due to Task 9 failure.")
             return

        self.log("=== Batch Analysis (Tasks 1-9) Completed Successfully ===")


    def create_task4_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 4")
        
        ttk.Label(tab, text="Input Time CSV (Processes all 'cliptime*.csv' in the selected file's folder):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t4_csv = ttk.Entry(tab, width=50)
        self.t4_csv.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_file(self.t4_csv)).grid(row=0, column=2)
        
        # Displaying target folders logic
        ttk.Label(tab, text="Processed Data (Auto-scans specific folder):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.t4_data = ttk.Entry(tab, width=50, state="disabled")
        self.t4_data.insert(0, "Auto-scans: ...[0_4800]_HPF8.0Hz_脈")
        self.t4_data.grid(row=1, column=1, padx=5, pady=5)
        
        # Information Note
        note_frame = ttk.LabelFrame(tab, text="Target Folder for Processed Data")
        note_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        ttk.Label(note_frame, text="Target: 0.05_0.02_0.07_1_3_[0_4800]_HPF8.0Hz_脈", justify=tk.LEFT).pack(anchor="w", padx=5, pady=2)

        ttk.Label(tab, text="Output Folder Name:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.t4_out = ttk.Entry(tab, width=50)
        self.t4_out.insert(0, os.path.join(self.base_dir, "時間区間比率_clipdata_NK04_HPF"))
        self.t4_out.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Button(tab, text="Run Task 4", command=self.run_task4).grid(row=4, column=1, pady=20)

    def run_task4(self):
        self.log("Running Task 4...")
        input_path = self.t4_csv.get()
        if not input_path:
            input_path = self.t1_csv.get()
            self.log(f"Task 4 Input empty, using Task 1 input: {input_path}")
            
        out_folder = self.t4_out.get()
        
        # 1. Input Time CSVs (Same as Task 1/2 logic)
        
        if not input_path or not os.path.exists(input_path):
             self.log(f"Error: Input path not found: {input_path}")
             return False
             
        if os.path.isfile(input_path):
            search_dir = os.path.dirname(input_path)
        else:
            search_dir = input_path
            
        cliptime_files = glob.glob(os.path.join(search_dir, "cliptime*.csv"))
        if not cliptime_files:
             self.log(f"No 'cliptime*.csv' files found in {search_dir}")
             return False
             
        # 2. Processed Data CSVs (Specific Folder for Task 4)
        target_folder_name = "0.05_0.02_0.07_1_3_[0_4800]_HPF8.0Hz_脈"
        folder_full_path = os.path.join(search_dir, target_folder_name)
        
        data_files = []
        if os.path.isdir(folder_full_path):
             # glob fails with brackets in path, so use os.listdir
            try:
                all_files = os.listdir(folder_full_path)
                found = [os.path.join(folder_full_path, f) for f in all_files if f.startswith("data_peak_pulse_") and f.endswith(".csv")]
                self.log(f"Found {len(found)} files in {target_folder_name}")
                data_files.extend(found)
            except Exception as e:
                self.log(f"Error reading directory {folder_full_path}: {e}")
        else:
             self.log(f"Warning: Folder not found: {folder_full_path}")
             
        if not data_files:
             self.log("No 'data_peak_pulse_*.csv' files found in Task 4 target folder.")
             return False
             
        # 3. Execution Loop
        success_count = 0
        error_count = 0
        
        for t_file in cliptime_files:
            for d_file in data_files:
                try:
                    # Task 4 uses lib_job2 but targets Task 4 output
                    lib_job2.main(t_file, d_file, out_folder)
                    success_count += 1
                except Exception as e:
                    self.log(f"  Error processing {os.path.basename(t_file)}: {e}")
                    error_count += 1
                    
        self.log(f"Task 4 Completed. Success: {success_count}, Errors: {error_count}")
        return success_count > 0

    def create_task5_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 5 (Copy)")
        
        ttk.Label(tab, text="Source Folder (Task 4 Output):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t5_src = ttk.Entry(tab, width=50)
        self.t5_src.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t5_src)).grid(row=0, column=2)
        
        ttk.Button(tab, text="Run Task 5", command=self.run_task5).grid(row=1, column=1, pady=20)

    def run_task5(self):
        self.log("Running Task 5...")
        src_root = self.t5_src.get()
        if not src_root:
            src_root = self.t4_out.get()
            self.log(f"Input empty, using Task 4 output: {src_root}")

        try:
            # Requirement: Copy all CSVs from Task 4's "20s_segments/data" to new folder "時間区間比率"
            src_20s = os.path.join(src_root, "20s_segments", "data")
            
            if not os.path.exists(src_20s):
                 self.log(f"[Error] Source folder not found: {src_20s}")
                 return False

            dest = os.path.join(os.path.dirname(os.path.abspath(src_root)), "時間区間比率")
            os.makedirs(dest, exist_ok=True)
            
            count = 0
            for f in os.listdir(src_20s):
                if f.endswith(".csv"):
                    shutil.copy2(os.path.join(src_20s, f), dest)
                    count += 1
            
            self.log(f"Copied {count} files from {src_20s} to {dest}")
            self.log("Task 5 Completed.")
            return True
        except Exception as e:
            self.log(f"Error in Task 5: {e}")
            return False

    def create_task6_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 6 (FFT)")
        
        ttk.Label(tab, text="Base Folder (Task 3 Output Parent):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t6_src = ttk.Entry(tab, width=50)
        self.t6_src.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t6_src)).grid(row=0, column=2)
        
        ttk.Button(tab, text="Run Task 6 (FFT Analysis)", command=self.run_task6).grid(row=1, column=1, pady=20)

    def run_task6(self):
        self.log("Running Task 6 (FFT)...")
        base = self.t6_src.get()
        
        if not base:
             # Try task 2 out's parent fallback
             t2_out = self.t2_out.get()
             if t2_out:
                 base = os.path.dirname(os.path.abspath(t2_out))
                 self.log(f"Input empty, trying Task 2 output parent: {base}")
        
        if not base or not os.path.exists(base):
            self.log(f"Error: Base folder not found: {base}")
            return False

        peak2_folder = os.path.join(base, "第２ピーク")
        mt_folder = os.path.join(base, "山")
        
        success = True
        
        # Run on Peak 2
        if os.path.exists(peak2_folder):
            self.log(f"Running FFT on {peak2_folder}...")
            try:
                lib_job5.run_fft_analysis(peak2_folder)
            except Exception as e:
                self.log(f"Error in FFT (Peak 2): {e}")
                success = False
        else:
            self.log(f"Skip: {peak2_folder} not found.")

        # Run on Mountain
        if os.path.exists(mt_folder):
            self.log(f"Running FFT on {mt_folder}...")
            try:
                lib_job5.run_fft_analysis(mt_folder)
            except Exception as e:
                self.log(f"Error in FFT (Mountain): {e}")
                success = False
        else:
             self.log(f"Skip: {mt_folder} not found.")
             
        if success:
            self.log("Task 6 Completed.")
        return success

    def create_task7_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 7 (Analysis)")
        
        ttk.Label(tab, text="Result Name (Prefix):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t7_name = ttk.Entry(tab, width=50)
        self.t7_name.insert(0, "NK04_カルテ時間_autothre")
        self.t7_name.grid(row=0, column=1, padx=5, pady=5)

        # 1. Peak 2 Result Folder (Task 6 Output)
        ttk.Label(tab, text="Peak 2 Result Folder (Task 6):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.t7_p2_res = ttk.Entry(tab, width=50)
        self.t7_p2_res.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t7_p2_res)).grid(row=1, column=2)

        # 2. Mountain Result Folder (Task 6 Output)
        ttk.Label(tab, text="Mountain Result Folder (Task 6):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.t7_mt_res = ttk.Entry(tab, width=50)
        self.t7_mt_res.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t7_mt_res)).grid(row=2, column=2)

        # 3. Time Ratio Folder (Task 5 Output)
        ttk.Label(tab, text="Time Ratio Folder (Task 5):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.t7_tr = ttk.Entry(tab, width=50)
        self.t7_tr.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t7_tr)).grid(row=3, column=2)


        
        ttk.Button(tab, text="Parameter Input", command=self.open_task7_config).grid(row=6, column=0, pady=20, padx=5)
        ttk.Button(tab, text="Run Analysis & Summarize", command=self.run_task7).grid(row=6, column=1, pady=20)

    def run_task7(self):
        self.log("Running Task 7 (Analysis)...")
        prefix = self.t7_name.get()
        p2_res_folder = self.t7_p2_res.get()
        mt_res_folder = self.t7_mt_res.get()
        tr_folder = self.t7_tr.get()
        
        # Validate inputs
        if not p2_res_folder or not os.path.exists(p2_res_folder):
             self.log(f"Warning: Peak 2 Result folder not found: {p2_res_folder}")
        if not mt_res_folder or not os.path.exists(mt_res_folder):
             self.log(f"Warning: Mountain Result folder not found: {mt_res_folder}")
        if not tr_folder or not os.path.exists(tr_folder):
             self.log(f"Warning: Time Ratio folder not found: {tr_folder}")

        # Retrieve parameters (or defaults)
        params = self.task7_params
        if params is None:
            self.log("No custom parameters set. Using strict defaults.")
            params = {
                "p1": {"interval": 2.0, "ratio_val": 70, "target_peak": 2, "split_start": 2, "split_end": 4},
                "p2": {"int_peak": 2.0, "int_sum": 1.0, "split_start": 2, "split_end": 3},
                "p3": [{
                    'mode_name': "1ファイルにつき1回判定",
                    'group_time': 2.0, 'top_n': 13, 'ratio1': "8/1", 'ratio2': "13/1", 'th1': 0.4, 'th2': 0.2,
                    'exclusion_th': 20.0, 'split_start': 2, 'split_end': 4
                }]
            }

        p1_res = None
        p2_res = None
        
        # Run Page 1 (Peak 2)
        p1_cfg = params.get("p1", {})
        if p2_res_folder and os.path.exists(p2_res_folder):
            target_p2 = os.path.join(p2_res_folder, "fft_csv1")
            if os.path.exists(target_p2):
                self.log(f"Running Page 1 (Peak 2) on {target_p2}...")
                files = [os.path.join(target_p2, f) for f in os.listdir(target_p2) if f.endswith(".csv")]
                if files:
                    lp1 = lib_job7.LogicPage1()
                    try:
                        int_w = float(p1_cfg.get("interval", 2.0))
                        rat_v = float(p1_cfg.get("ratio_val", 70))
                        t_peak = int(p1_cfg.get("target_peak", 2))
                        s_s = int(p1_cfg.get("split_start", 2))
                        s_e = int(p1_cfg.get("split_end", 4))
                        
                        p1_res = lp1.run_analysis(files, "Res_P1", prefix, int_w, rat_v, t_peak, s_s, s_e, base_output_dir=self.base_dir)
                    except Exception as e:
                        self.log(f"Error in Page 1: {e}")
                else:
                     self.log("No CSV files found in Peak 2 fft_csv1 folder.")
            else:
                 self.log(f"fft_csv1 folder not found in {p2_res_folder}")
        else:
            self.log("Skipping Page 1 (Peak 2).")
            
        # Run Page 2 (Mountain)
        p2_cfg = params.get("p2", {})
        if mt_res_folder and os.path.exists(mt_res_folder):
            target_mt = os.path.join(mt_res_folder, "fft_csv1")
            if os.path.exists(target_mt):
                self.log(f"Running Page 2 (Convex) on {target_mt}...")
                files = [os.path.join(target_mt, f) for f in os.listdir(target_mt) if f.endswith(".csv")]
                if files:
                    lp2 = lib_job7.LogicPage2()
                    try:
                        int_peak = float(p2_cfg.get("int_peak", 2.0))
                        int_sum = float(p2_cfg.get("int_sum", 1.0))
                        s_s = int(p2_cfg.get("split_start", 2))
                        s_e = int(p2_cfg.get("split_end", 3))
                        
                        p2_res = lp2.run_analysis(files, "Res_P2", prefix, int_peak, int_sum, s_s, s_e, base_output_dir=self.base_dir)
                    except Exception as e:
                        self.log(f"Error in Page 2: {e}")
                else:
                     self.log("No CSV files found in Mountain fft_csv1 folder.")
            else:
                 self.log(f"fft_csv1 folder not found in {mt_res_folder}")
        else:
             self.log("Skipping Page 2 (Mountain).")

        # Run Page 3 (Time Ratio)
        p3_configs = params.get("p3", [])
        success = True
        p3_summaries_list = []
        
        if tr_folder and os.path.exists(tr_folder):
            files = [os.path.join(tr_folder, f) for f in os.listdir(tr_folder) if f.endswith(".csv")]
            if files:
                lp3 = lib_job7.LogicPage3()
                
                for cfg in p3_configs:
                    mode_name = cfg['mode_name']
                    self.log(f"Running Page 3 Mode: {mode_name}")
                    try:
                        # Ensure common ID split settings are respected if not in cfg
                        if 'split_end' not in cfg: cfg['split_end'] = 4
                        
                        p3_res = lp3.run_analysis_mode(files, "Res_P3", prefix, cfg, base_output_dir=self.base_dir)
                        
                        suffix = mode_name.replace("(", "_").replace(")", "").replace(" ", "")
                        save_name = f"Final_Summary_{prefix}_{suffix}.xlsx"
                        lib_job7.create_summary_excel(p1_res, p2_res, p3_res, save_name)
                        if p3_res is not None:
                            p3_summaries_list.append(p3_res)
                        
                    except Exception as e:
                        self.log(f"Error in Page 3 ({mode_name}): {e}")
                        import traceback
                        traceback.print_exc()
            else:
                self.log("No CSV files found in Time Ratio folder.")
        else:
            self.log("Skipping Page 3 (Time Ratio).")
            
        # Save Final Summary (Excel)
        if p1_res is not None or p2_res is not None or p3_summaries_list:
             # Merge all p3 results
             p3_merged = pd.DataFrame()
             for s in p3_summaries_list:
                 if p3_merged.empty: p3_merged = s
                 else: p3_merged = pd.concat([p3_merged, s], ignore_index=True)
             
             final_name = f"Final_Summary_{prefix}.xlsx"
             
             # Try saving in current directory first (or user specified?)
             # The lib currently saves to the path passed as arg.
             # We want to save it to ONE file.
             # Let's save it to the script directory for now, or a "Results" folder?
             # User asked for "Final_Summary_NK05_..." naming.
             # We will use the prefix.
             
             final_path = os.path.join(self.base_dir, final_name)
             lib_job7.create_summary_excel(p1_res, p2_res, p3_merged, final_path)
             self.log(f"Tasks 7 Completed. Summary saved to: {final_path}")
             
        return True

    def create_task8_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 8 (Kawanobe & DifTime)")
        
        # F1 Folder
        ttk.Label(tab, text="F1 Folder (Positive/Data):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.t8_f1 = ttk.Entry(tab, width=50)
        self.t8_f1.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t8_f1)).grid(row=0, column=2)
        
        # F2 Folder
        ttk.Label(tab, text="F2 Folder (Negative/Reverse):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.t8_f2 = ttk.Entry(tab, width=50)
        self.t8_f2.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t8_f2)).grid(row=1, column=2)


        
        ttk.Button(tab, text="Run Task 8", command=self.run_task8).grid(row=4, column=1, pady=20)

    def run_task8(self):
        self.log("Running Task 8 (Kawanobe Verification & DifTime Stats)...")
        f1_dir = self.t8_f1.get()
        f2_dir = self.t8_f2.get()

        # --- Auto-fill from Task 2 Output if empty ---
        t2_out_path = self.t2_out.get()

        if not f1_dir and t2_out_path:
            f1_dir = os.path.join(t2_out_path, "20s_segments", "data")
            self.log(f"[Task 8] Auto-detected F1 Folder: {f1_dir}")

        if not f2_dir and t2_out_path:
            f2_dir = os.path.join(t2_out_path, "20s_segments", "peak_reverse")
            self.log(f"[Task 8] Auto-detected F2 Folder: {f2_dir}")
        # ---------------------------------------------
        
        # User specified that F1 is the same as Dif Time Source, and F2 is same as Peak Reverse Source
        dif_src = f1_dir
        pr_src = f2_dir

        
        if not f1_dir or not os.path.exists(f1_dir):
             self.log("Error: F1 Folder invalid.")
             return False
        if not f2_dir or not os.path.exists(f2_dir):
             self.log("Error: F2 Folder invalid.")
             return False

        # Output 1: Kawanobe Results (Script Dir)
        kawanobe_out = os.path.join(self.base_dir, "kawanobe_results")
        os.makedirs(kawanobe_out, exist_ok=True)
        
        # Output 2: Append to Final Summary (Task 7)
        # Find latest summary in base_dir created by Task 7
        prefix = self.t7_name.get()
        # The summary filename is constructed in Task 7
        # pattern: Final_Summary_{prefix}_*.xlsx
        summary_file = None
        candidates = glob.glob(os.path.join(self.base_dir, f"Final_Summary_{prefix}_*.xlsx"))
        if candidates:
            # Sort by time?
            summary_file = max(candidates, key=os.path.getmtime)
            self.log(f"Targeting Summary File: {summary_file}")
        else:
            self.log("Warning: Task 7 Summary File not found. Stats will only be printed/saved locally.")

        f1_files = glob.glob(os.path.join(f1_dir, "*.csv"))
        f2_files_all = glob.glob(os.path.join(f2_dir, "*.csv"))
        # Map F2
        f2_map = {}
        for f in f2_files_all:
             fid = lib_job8.get_id_from_filename(os.path.basename(f))
             if fid: f2_map[fid] = f
        
        updates = {} # ID -> stats
        
        for f1 in f1_files:
            fid = lib_job8.get_id_from_filename(os.path.basename(f1))
            if not fid: continue
            
            if fid in f2_map:
                f2 = f2_map[fid]
                self.log(f"  Processing {fid}...")
                
                # 1. Run Kawanobe -> Get Intervals
                intervals, res_path = lib_job8.run_kawanobe_logic(f1, f2, kawanobe_out)
                
                if intervals:
                    # 2. Clip & Calc Stats
                    # Need to find source file in dif_src / pr_src
                    # Source file name pattern: clipdata_{ID}_{no1}_{no2}_... 
                    # Use lib_job8.extract_id_detailed to match or simple startswith
                    
                    dif_stats = []
                    if dif_src:
                         # Find file starting with clipdata_{fid} or containing {fid}
                         # This is tricky because FID is NK05_3_1 but file might be clipdata_NK05_3_1...
                         # Let's search for *{fid}*.csv
                         cand = [x for x in os.listdir(dif_src) if fid in x and x.endswith(".csv")]
                         if cand:
                             # Pick "best" candidate? Usually just one per ID in that folder
                             # If multiple, pick first?
                             src_f = os.path.join(dif_src, cand[0])
                             dif_stats = lib_job8.process_clipped_stats(intervals, src_f, target_col="dif_time")
                    
                    pr_stats = []
                    if pr_src:
                         cand = [x for x in os.listdir(pr_src) if fid in x and x.endswith(".csv")]
                         if cand:
                             src_f = os.path.join(pr_src, cand[0])
                             pr_stats = lib_job8.process_clipped_stats(intervals, src_f, target_col="dif_time") # PeakReverse CSV also uses 'dif_time' column usually
                    
                    updates[fid] = {
                        'intervals': intervals,
                        'dif_stats': dif_stats,
                        'pr_stats': pr_stats
                    }
        
        # 3. Update Excel
        if summary_file and updates:
            lib_job8.update_summary_excel(summary_file, updates)
            
        self.log(f"Task 8 Completed. Processed {len(updates)} IDs.")
        return True

    def create_task9_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Task 9 (Karte Data)")
        self.t8_out = ttk.Entry(tab, width=50)
        self.t8_out.insert(0, os.path.join(self.base_dir, "カルテ作成に必要なデータ"))
        self.t8_out.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(tab, text="Browse", command=lambda: self.select_folder(self.t8_out)).grid(row=0, column=2)
        
        ttk.Button(tab, text="Run Task 9 (Collect Data)", command=self.run_task9).grid(row=1, column=1, pady=20)

    def run_task9(self):
        self.log("Running Task 9 (Collecting Data for Clinical Chart)...")
        target_root = self.t8_out.get()
        if not target_root:
            target_root = "カルテ作成に必要なデータ"
        
        target_karte = os.path.join(target_root, "カルテ")
        target_fft_20s = os.path.join(target_root, "FFT", "20s")
        target_fft_60s = os.path.join(target_root, "FFT", "60s")
        
        os.makedirs(target_karte, exist_ok=True)
        os.makedirs(target_fft_20s, exist_ok=True)
        os.makedirs(target_fft_60s, exist_ok=True)
        
        # 1. Copy Task 1 Images (clipdata_NK05_raw etc)
        # Look for 20s_segments/raw/graph OR 20s_segments/data/graph
        t1_out = self.t1_out.get()
        if t1_out and os.path.exists(t1_out):
            src_graph = os.path.join(t1_out, "20s_segments", "raw", "graph")
            if not os.path.exists(src_graph):
                src_graph = os.path.join(t1_out, "20s_segments", "data", "graph")
            
            if os.path.exists(src_graph):
                folder_name = os.path.basename(t1_out)
                dest = os.path.join(target_karte, folder_name)
                os.makedirs(dest, exist_ok=True)
                self.copy_files(src_graph, dest)
                self.log(f"Copied Task 1 images from {src_graph} to {dest}")
            else:
                self.log(f"Warning: Task 1 graph folder not found in {t1_out}")
        
        # 2. Copy Task 2 Images (clipdata_NK05_HPF etc)
         # Look for 20s_segments/data/graph OR 20s_segments/raw/graph
        t2_out = self.t2_out.get()
        if t2_out and os.path.exists(t2_out):
            src_graph = os.path.join(t2_out, "20s_segments", "data", "graph")
            if not os.path.exists(src_graph):
                src_graph = os.path.join(t2_out, "20s_segments", "raw", "graph")

            if os.path.exists(src_graph):
                folder_name = os.path.basename(t2_out)
                dest = os.path.join(target_karte, folder_name)
                os.makedirs(dest, exist_ok=True)
                self.copy_files(src_graph, dest)
                self.log(f"Copied Task 2 images from {src_graph} to {dest}")
            else:
                self.log(f"Warning: Task 2 graph folder not found in {t2_out}")

        # 3. Copy Final Summary Excel
        # Search in current dir? Or where Task 7 saved it?
        # Pattern: Final_Summary_*.xlsx
        excel_files = glob.glob("Final_Summary_*.xlsx")
        if excel_files:
            for f in excel_files:
                shutil.copy2(f, target_karte)
                self.log(f"Copied {f} to {target_karte}")
        else:
            self.log("Warning: Final_Summary_*.xlsx not found.")
            
        # 4. Copy FFT Images (Peak 2 -> 20s) - FROM TASK 7 Output "graphs"
        # Reconstruct Task 7 P1 Output Folder Name
        prefix = self.t7_name.get()
        params = self.task7_params
        if params is None:
             params = {
                "p1": {"interval": 2.0, "ratio_val": 70, "target_peak": 2},
                "p2": {"int_peak": 2.0, "int_sum": 1.0}
            }
        
        # P1 (Peak 2) Folder Name Reconstruction
        p1_cfg = params.get("p1", {})
        try:
            int_w = float(p1_cfg.get("interval", 2.0))
            rat_v = float(p1_cfg.get("ratio_val", 70))
            t_peak = int(p1_cfg.get("target_peak", 2))
            
            interval_str = str(int_w).replace('.', '_')
            ratio_str = str(int(rat_v))
            peak_str = f"peak{t_peak}"
            
            save_folder_name_base = f"Res_P1_range{interval_str}_ratio{ratio_str}_{peak_str}"
            p1_folder_name = f"{prefix}_{save_folder_name_base}" if prefix else save_folder_name_base
            
            p1_graphs_path = os.path.join(self.base_dir, p1_folder_name, "graphs")
            
            if os.path.exists(p1_graphs_path):
                 self.copy_files(p1_graphs_path, target_fft_20s)
                 self.log(f"Copied Peak 2 FFT images from {p1_graphs_path} to {target_fft_20s}")
            else:
                 self.log(f"Warning: Peak 2 FFT graph folder not found: {p1_graphs_path}")
        except Exception as e:
            self.log(f"Error resolving Peak 2 Task 7 folder: {e}")

        # 5. Copy FFT Images (Mountain -> 60s) - FROM TASK 7 Output "graphs"
        # P2 (Mountain) Folder Name Reconstruction
        p2_cfg = params.get("p2", {})
        try:
            int_peak = float(p2_cfg.get("int_peak", 2.0))
            int_sum = float(p2_cfg.get("int_sum", 1.0))
            
            i_p_str = str(int_peak).replace('.', '_')
            i_s_str = str(int_sum).replace('.', '_')
            
            save_folder_name_base = f"Res_P2_peakRange{i_p_str}_sumRange{i_s_str}"
            p2_folder_name = f"{prefix}_{save_folder_name_base}" if prefix else save_folder_name_base
            

            
            p2_graphs_path = os.path.join(self.base_dir, p2_folder_name, "graphs")
            
            if os.path.exists(p2_graphs_path):
                 self.copy_files(p2_graphs_path, target_fft_60s)
                 self.log(f"Copied Mountain FFT images from {p2_graphs_path} to {target_fft_60s}")
            else:
                 self.log(f"Warning: Mountain FFT graph folder not found: {p2_graphs_path}")
        except Exception as e:
            self.log(f"Error resolving Mountain Task 7 folder: {e}")

        self.log("Task 9 Completed.")
        return True

    def copy_files(self, src, dest):
        for f in os.listdir(src):
            s = os.path.join(src, f)
            if os.path.isfile(s):
                shutil.copy2(s, dest)

if __name__ == "__main__":
    app = IntegratedRunnerApp()
    app.mainloop()
