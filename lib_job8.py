
import pandas as pd
import numpy as np
import os
import glob
import re
import statistics
import shutil
from openpyxl import load_workbook

# --- Kawanobe 定数 (ms単位に変更) ---
MATCH_THRESHOLD = 150      # 0.15s -> 150ms
BAD_ZONE_MARGIN = 100      # 0.1s -> 100ms
MIN_USABLE_DURATION = 4000 # 4.0s -> 4000ms

def get_id_from_filename(filename):
    match = re.search(r'(NK|KA)\d+_\d+_\d+', filename)
    if match:
        return match.group(0)
    return None

def extract_id_detailed(filename):
    # ソースファイル (例: clipdata_NK05_3_1_...) と照合するために詳細IDを抽出
    # 標準ID形式は NK05_3_1 と仮定
    base = os.path.splitext(os.path.basename(filename))[0]
    return base

# --- Kawanobe ロジック ---
def run_kawanobe_logic(f1_path, f2_path, output_dir):
    # Helper to get timestamp values
    def get_timestamps(df, label):
        # 優先: peak_timestamp
        if 'peak_timestamp' in df.columns:
            return df['peak_timestamp'].dropna().values
        # 次点: timestamp (peak_reverseなどの場合)
        if 'timestamp' in df.columns:
            return df['timestamp'].dropna().values
        # フォールバック: カラム名に timestamp が含まれるもの
        cols = [c for c in df.columns if 'timestamp' in str(c).lower()]
        if cols:
            return df[cols[0]].dropna().values
        # 最終手段: インデックス4 (Data+Peakの場合) or 1 (PeakReverse).. 危険なのでカラム探索のみ
        return None

    try:
        df1 = pd.read_csv(f1_path)
        f1_times = get_timestamps(df1, "F1")
        if f1_times is None:
             # Legacy fallback: try iloc 4 (peak_timestamp expected position in standard)
             if df1.shape[1] > 4: f1_times = df1.iloc[:, 4].dropna().values
             else: return None, [] # Cannot find timestamps
        
        df2 = pd.read_csv(f2_path)
        f2_times = get_timestamps(df2, "F2")
        if f2_times is None:
             if df2.shape[1] > 1:
                # Try index 1 (peak_reverse standard?)
                # But safer to return empty if not found
                pass
        
        if f2_times is None or len(f2_times) == 0: return None, []
        f2_times = np.sort(f2_times)
        
        if len(f2_times) == 0: return None, []

        file_id = get_id_from_filename(os.path.basename(f1_path))
        if not file_id: file_id = os.path.splitext(os.path.basename(f1_path))[0]

        # Positive 軸
        p_base, p_closest, p_diff = [], [], []
        for t1 in f1_times:
            idx = np.searchsorted(f2_times, t1)
            candidates = []
            if idx < len(f2_times): candidates.append(idx)
            if idx > 0: candidates.append(idx - 1)
            
            if not candidates:
                c_t2, d = np.nan, np.nan
            else:
                closest_idx = min(candidates, key=lambda i: abs(f2_times[i] - t1))
                c_t2 = f2_times[closest_idx]
                d = abs(t1 - c_t2)
            p_base.append(t1)
            p_closest.append(c_t2)
            p_diff.append(d)
            
        df_p = pd.DataFrame({'ID': [file_id]*len(p_base), 'positive_peak_timestamp': p_base, 'closest_negative_peak_timestamp': p_closest, 'abs_diff_positive_axis': p_diff})
        
        # Negative 軸
        n_base, n_closest, n_diff = [], [], []
        f1_sorted = np.sort(f1_times)
        for t2 in f2_times:
            idx = np.searchsorted(f1_sorted, t2)
            candidates = []
            if idx < len(f1_sorted): candidates.append(idx)
            if idx > 0: candidates.append(idx - 1)
            
            if not candidates:
                c_t1, d = np.nan, np.nan
            else:
                closest_idx = min(candidates, key=lambda i: abs(f1_sorted[i] - t2))
                c_t1 = f1_sorted[closest_idx]
                d = abs(t2 - c_t1)
            n_base.append(t2)
            n_closest.append(c_t1)
            n_diff.append(d)
        
        df_n = pd.DataFrame({'negative_peak_timestamp': n_base, 'closest_positive_peak_timestamp': n_closest, 'abs_diff_negative_axis': n_diff})
        
        # 検証 (Validation)
        p_closest_set = set(x for x in p_closest if not np.isnan(x))
        n_closest_set = set(x for x in n_closest if not np.isnan(x))
        
        p_is_in = np.array([x in n_closest_set for x in p_base])
        p_diff_ok = np.array([(d < MATCH_THRESHOLD if not np.isnan(d) else False) for d in p_diff])
        p_valid = p_is_in & p_diff_ok
        
        n_is_in = np.array([x in p_closest_set for x in n_base])
        n_diff_ok = np.array([(d < MATCH_THRESHOLD if not np.isnan(d) else False) for d in n_diff])
        n_valid = n_is_in & n_diff_ok
        
        df_p_valid = pd.DataFrame({'time': p_base, 'valid': p_valid})
        df_n_valid = pd.DataFrame({'time': n_base, 'valid': n_valid})
        
        df_merged = pd.concat([df_p_valid, df_n_valid]).sort_values(by='time')
        sorted_times = df_merged['time'].values
        valid_results = df_merged['valid'].values
        
        # 検証結果の出力カラム
        df_p_val_out = pd.DataFrame({'': [np.nan]*len(p_base), 'F1_Timestamp_Validation': p_base, 'In_F2_Results': p_is_in, 'Diff_lt_150ms_F1': p_diff_ok})
        df_n_val_out = pd.DataFrame({' ': [np.nan]*len(n_base), 'F2_Timestamp_Validation': n_base, 'In_F1_Results': n_is_in, 'Diff_lt_150ms_F2': n_diff_ok})
        df_sorted_val = pd.DataFrame({'  ': [np.nan]*len(sorted_times), 'Sorted_Timestamp': sorted_times, 'Validation_Result': valid_results})

        # 有効区間の抽出 (Intervals)
        usable_intervals = []
        if len(sorted_times) > 0:
            bad_zones = []
            i = 0
            n = len(sorted_times)
            while i < n:
                if not valid_results[i]:
                    start_idx = i
                    while i < n and not valid_results[i]: i += 1
                    end_idx = i - 1
                    t_start_bad = sorted_times[start_idx] - BAD_ZONE_MARGIN
                    t_end_bad = sorted_times[end_idx] + BAD_ZONE_MARGIN
                    bad_zones.append((t_start_bad, t_end_bad))
                else:
                    i += 1
            
            file_start = sorted_times[0]
            file_end = sorted_times[-1]
            current_valid_start = file_start
            
            for bad_start, bad_end in bad_zones:
                if bad_start > current_valid_start:
                    seg_end = bad_start
                    if (seg_end - current_valid_start) >= MIN_USABLE_DURATION:
                        usable_intervals.append((current_valid_start, seg_end))
                if bad_end > current_valid_start:
                    current_valid_start = bad_end
            
            if current_valid_start < file_end:
                if (file_end - current_valid_start) >= MIN_USABLE_DURATION:
                    usable_intervals.append((current_valid_start, file_end))

        # 出力の構築
        u_starts, u_ends = [], []
        if usable_intervals:
            for s, e in usable_intervals:
                u_starts.append("start" if s == sorted_times[0] else s)
                u_ends.append("end" if e == sorted_times[-1] else e)
        
        df_intervals = pd.DataFrame({'   ': [np.nan]*len(u_starts), 'Usable_Start_Timestamp': u_starts, 'Usable_End_Timestamp': u_ends})
        
        result_df = pd.concat([df_p, df_n, df_p_val_out, df_n_val_out, df_sorted_val, df_intervals], axis=1)
        
        out_name = f"kawanobe_{file_id}.csv"
        out_path = os.path.join(output_dir, out_name)
        result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        
        return usable_intervals, out_path

    except Exception as e:
        print(f"Kawanobe Logic Error: {e}")
        return None, None

def calculate_stats(data):
    if len(data) < 2: return np.nan, np.nan, np.nan
    sigma = statistics.pstdev(data)
    variance = statistics.pvariance(data)
    diffs = np.diff(data)
    sum_diffs = np.sum(diffs**2)
    rmssd = np.sqrt((1/(len(data)-1))*sum_diffs) * 1000
    return sigma, variance, rmssd

def process_clipped_stats(intervals, source_file, target_col="dif_time"):
    if not source_file or not os.path.exists(source_file): return []
    try:
        df = pd.read_csv(source_file)
        if target_col not in df.columns: return []
        # dfに 'time' 列があるか確認 (標準では 'time' と 'dif_time' がある)
        # 実際にTask 2のアウトプットには通常 'time' 列がある
        # 'time' 列が存在するか検証
        time_col = None
        time_col = None
        for c in df.columns:
            # 優先的に timestamp を探す
            if "peak_timestamp" in c.lower():
                 time_col = c
                 break
            if "timestamp" in c.lower():
                 time_col = c
                 # breakしない (peak_timestampの方が良いかもしれないので)
                 # しかしイテレーション順序による。peak_timestampを先に見つけたいなら上記ifでOK
        
        # 見つからなかった場合、既存ロジック(time)へ
        if not time_col:
             for c in df.columns:
                if "peak_time" in c.lower():
                     time_col = c
                     break
                if "time" in c.lower() and "dif" not in c.lower():
                     time_col = c
        
        if not time_col: 
             # フォールバック
             if 'time' in df.columns: time_col = 'time'
             else: return []

        stats = []
        for start, end in intervals:
            # "start" と "end" の文字列を処理
            # 文字列 "start" なら 0 または最初のタイムスタンプを使用
            # ただし、ここで渡される intervals は上記のロジックからの (float, float) である。
            
            # フィルタリング
            mask = (df[time_col] >= start) & (df[time_col] <= end)
            subset = df.loc[mask, target_col].dropna()

            if not subset.empty:
                # 先頭の値が閾値以上なら除外
                if np.abs(subset.iloc[0]-subset.iloc[1]) >= (MATCH_THRESHOLD / 1000):
                    subset = subset.iloc[1:]
                
                print(f"Usable Data Indices (Time: {start} ~ {end}): {subset.index.tolist()}")
            
            if len(subset) > 1:
                vals = [float(x) for x in subset]
                s, v, r = calculate_stats(vals)
                stats.append((s, v, r))
            else:
                 stats.append((np.nan, np.nan, np.nan))
        return stats
    except Exception as e:
        print(f"Stats Calcuration Error: {e}")
        return []

def update_summary_excel(final_summary_path, updates):
    """
    updates: dict { 'ID': { 'intervals': [(s,e)...], 'dif_stats': [(s,v,r)...], 'pr_stats': [(s,v,r)...] } }
    """
    if not os.path.exists(final_summary_path):
        print("Final Summary file not found.")
        return

    try:
        df = pd.read_excel(final_summary_path, engine='openpyxl')
        
        # カラムの準備
        # G..P は 区間用 (5枠)
        # Q..U は Dif Sigma
        # V..Z は Dif Variance
        # AA..AE は Dif RMSSD
        # AF..AJ は PR Sigma
        # AK..AO は PR Variance
        # AP..AT は PR RMSSD
        
        # pandasで実際の列文字にマッピングするのは難しいので、列名を使用します。
        # 動的な名前を生成します。
        
        # カラムのクリア/作成
        cols_map = {
            'Usable_Start_': 5, 'Usable_End_': 5,
            'Dif_Sigma_': 5, 'Dif_Variance_': 5, 'Dif_RMSSD_': 5,
            'PR_Sigma_': 5, 'PR_Variance_': 5, 'PR_RMSSD_': 5
        }
        
        # 値を設定するヘルパー関数
        def set_val(row_idx, col_name, val):
            df.at[row_idx, col_name] = val

        # カラムが存在することを確認
        needed_cols = []
        for i in range(1, 6):
            needed_cols.extend([
                f"Usable_Start_{i}", f"Usable_End_{i}",
                f"dif_time_sigma_{i}", f"dif_time_variance_{i}", f"dif_time_rmssd[ms]_{i}",
                f"peak_reverse_sigma_{i}", f"peak_reverse_variance_{i}", f"peak_reverse_rmssd[ms]_{i}"
            ])
        
        for col in needed_cols:
            if col not in df.columns:
                df[col] = np.nan

            
        for idx, row in df.iterrows():
            fid = str(row['ID'])
            if fid in updates:
                data = updates[fid]
                intervals = data.get('intervals', [])
                dif_stats = data.get('dif_stats', [])
                pr_stats = data.get('pr_stats', [])
                
                # Fill Intervals
                for i in range(5):
                    s_col = f"Usable_Start_{i+1}"
                    e_col = f"Usable_End_{i+1}"
                    if i < len(intervals):
                        set_val(idx, s_col, intervals[i][0])
                        set_val(idx, e_col, intervals[i][1])
                    else:
                        set_val(idx, s_col, np.nan)
                        set_val(idx, e_col, np.nan)

                # Fill Dif Stats (Positive Axis)
                for i in range(5):
                    if i < len(dif_stats):
                        set_val(idx, f"dif_time_sigma_{i+1}", dif_stats[i][0])
                        set_val(idx, f"dif_time_variance_{i+1}", dif_stats[i][1])
                        set_val(idx, f"dif_time_rmssd[ms]_{i+1}", dif_stats[i][2])
                    else:
                        set_val(idx, f"dif_time_sigma_{i+1}", np.nan)
                        set_val(idx, f"dif_time_variance_{i+1}", np.nan)
                        set_val(idx, f"dif_time_rmssd[ms]_{i+1}", np.nan)

                # Fill PR Stats (Negative Axis)
                for i in range(5):
                    if i < len(pr_stats):
                        set_val(idx, f"peak_reverse_sigma_{i+1}", pr_stats[i][0])
                        set_val(idx, f"peak_reverse_variance_{i+1}", pr_stats[i][1])
                        set_val(idx, f"peak_reverse_rmssd[ms]_{i+1}", pr_stats[i][2])
                    else:
                        set_val(idx, f"peak_reverse_sigma_{i+1}", np.nan)
                        set_val(idx, f"peak_reverse_variance_{i+1}", np.nan)
                        set_val(idx, f"peak_reverse_rmssd[ms]_{i+1}", np.nan)

        # Save back
        df.to_excel(final_summary_path, index=False)
        print("Updated Final Summary.")
        
    except Exception as e:
        print(f"Error updating excel: {e}")

