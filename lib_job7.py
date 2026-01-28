
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import math
import traceback
import statistics
import numpy as np
from scipy.signal import find_peaks
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from collections import Counter

# --- 共通ヘルパー ---

def validate_peak_data(values, min_th=20, max_th=300):
    if len(values) == 0:
        return 0, 0, 0 # valid, sigma3, split_invalid

    std = values.std()
    sigma3 = 3 * std
    
    # --- DEBUG START ---
    print(f"  [DEBUG] Global 3Sigma: {sigma3:.2f} (Range: {min_th}-{max_th})")
    # --- DEBUG END ---

    is_valid = 1
    if sigma3 <= min_th or sigma3 >= max_th:
        is_valid = 0
        print("  [DEBUG] -> Global Invalid")
        
    split_invalid_count = 0
    chunks = np.array_split(values, 5)
    for i, c in enumerate(chunks):
        if len(c) == 0: continue
        c_std = c.std()
        c_sigma = 3 * c_std
        
        chunk_status = "Valid"
        if c_sigma <= min_th or c_sigma >= max_th:
            split_invalid_count += 1
            chunk_status = "Invalid"
        
        # --- DEBUG START ---
        print(f"    [DEBUG] Chunk {i}: 3Sigma={c_sigma:.2f} -> {chunk_status}")
        # --- DEBUG END ---
            
    if is_valid == 1:
        if split_invalid_count >= 2:
            is_valid = 0
            print(f"  [DEBUG] -> Re-evaluated as Invalid (Split Invalid Count: {split_invalid_count}/5)")
            
    return is_valid, sigma3, split_invalid_count

def setup_japanese_font():
    font_path = None
    font_options = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ]
    # 速度と信頼性のために、まず一般的なシステムの場所を確認する
    for path in font_options:
        if os.path.exists(path):
            font_path = path
            break
    
    # 一般的なパスで見つからない場合、全システムフォントを検索（遅くなる）
    if not font_path:
        for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
            if 'Meiryo' in font or 'Hiragino' in font or 'Noto Sans CJK JP' in font:
                font_path = font
                break

    if font_path:
        try:
            # フォントマネージャーに明示的にフォントを追加 - ファミリー名を正しく解決するために重要
            fm.fontManager.addfont(font_path) 
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"日本語フォント設定: {prop.get_name()} ({font_path})")
        except Exception as e:
            print(f"フォント設定エラー: {e}")
            pass

def extract_id(file_name, start_idx, end_idx):
    try:
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        parts = base_name.split('_')
        start_zero_idx = start_idx - 1
        end_zero_idx = end_idx 
        if start_zero_idx < 0: start_zero_idx = 0
        if end_zero_idx > len(parts): end_zero_idx = len(parts)
        if start_zero_idx >= len(parts) or start_zero_idx >= end_zero_idx:
            return base_name 
        extracted_parts = parts[start_zero_idx:end_zero_idx]
        if not extracted_parts: return base_name
        return "_".join(extracted_parts)
    except Exception:
        return os.path.splitext(os.path.basename(file_name))[0]

def is_convex_upwards(y_values):
    if len(y_values) < 3: return False
    max_idx = np.argmax(y_values)
    if max_idx == 0 or max_idx == len(y_values) - 1: return False
    is_increasing = all(y_values[i] <= y_values[i+1] for i in range(max_idx))
    is_decreasing = all(y_values[i] >= y_values[i+1] for i in range(max_idx, len(y_values) - 1))
    return is_increasing and is_decreasing

# --- ページ3 ヘルパー ---

def load_csv_page3(path):
    df = pd.read_csv(path)
    # 元の必須列を確認
    if "peak_time" in df.columns and "peak_value" in df.columns:
        peak_time_numeric = pd.to_numeric(df["peak_time"], errors='coerce')
        peak_value_numeric = pd.to_numeric(df["peak_value"], errors='coerce')
        timestamp_numeric = pd.to_numeric(df["timestamp"], errors='coerce') if "timestamp" in df.columns else pd.Series([0]*len(df))
        
        df = pd.DataFrame({
            "time": peak_time_numeric,
            "timestamp": timestamp_numeric,
            "peak_value": peak_value_numeric,
            "dif_time": pd.to_numeric(df["dif_time"], errors='coerce') if "dif_time" in df.columns else pd.Series([0]*len(df)),
        })
        df = df.dropna(subset=["time", "peak_value"]).reset_index(drop=True)
        if df.empty: raise ValueError("No valid data.")
        return df
    else:
        # フォールバックまたはエラー
        raise ValueError("Missing required columns: peak_time, peak_value")

def cal_peakvalue_ratio_page3(peak_values, top_n_peaks, ratios_to_calc):
    results = {f'ratio_{num}_{den}': np.nan for num, den in ratios_to_calc}
    results['top_peaks'] = [np.nan] * top_n_peaks
    if peak_values.empty: return results
    sorted_peaks = peak_values.sort_values(ascending=False).reset_index(drop=True)
    num_peaks = len(sorted_peaks)
    peaks_to_store = min(num_peaks, top_n_peaks)
    results['top_peaks'][:peaks_to_store] = sorted_peaks.head(peaks_to_store).tolist()
    if num_peaks == 0: return results
    for num, den in ratios_to_calc:
        num_idx, den_idx = num - 1, den - 1
        if num_idx < num_peaks and den_idx < num_peaks:
            denominator_peak = sorted_peaks.iloc[den_idx]
            if denominator_peak != 0:
                numerator_peak = sorted_peaks.iloc[num_idx]
                results[f'ratio_{num}_{den}'] = numerator_peak / denominator_peak
    return results

def analyze_group_page3(group_df, top_n_peaks, ratios_to_calc, auto_threshold1=0.6, auto_threshold2=0.3, 
                        use_average_center=False, manual_center_idx=0, center_offset=3,
                        use_global_mode=False, global_center_peak_idx=0):
    analysis_results = cal_peakvalue_ratio_page3(group_df['peak_value'], top_n_peaks, ratios_to_calc)
    count = len(group_df)
    flat_results = {'counts': count}
    flat_results['max_peak_value'] = group_df['peak_value'].max() if not group_df['peak_value'].empty else 0

    for num, den in ratios_to_calc:
        flat_results[f'peak_{num}/peak_{den}'] = analysis_results.get(f'ratio_{num}_{den}', np.nan)
    for i, peak in enumerate(analysis_results.get('top_peaks', [])):
        flat_results[f'peak_{i+1}'] = peak
        
    target_nums = range(2, 14)
    all_ratios_req = [(n, 1) for n in target_nums]
    full_analysis = cal_peakvalue_ratio_page3(group_df['peak_value'], 13, all_ratios_req)
    
    all_valid_ratios = []
    ratio_map = {}
    for n in target_nums:
        key = f'ratio_{n}_1'
        val = full_analysis.get(key, np.nan)
        col_name = f'peak_{n}/peak_1'
        if col_name not in flat_results: flat_results[col_name] = val
        if not pd.isna(val):
            all_valid_ratios.append(val)
            ratio_map[n] = val

    median_val = np.nan
    median_idx = -1
    calc_point_auto = 0
    point_auto_n_minus_3 = 0
    point_auto_n_plus_3 = 0
    
    if all_valid_ratios:
        median_val = np.median(all_valid_ratios)
        best_diff = float('inf')
        for n, v in ratio_map.items():
            diff = abs(v - median_val)
            if diff < best_diff:
                best_diff = diff
                median_idx = n
        
        idx_minus_3 = median_idx - 3
        if idx_minus_3 in ratio_map and ratio_map[idx_minus_3] >= auto_threshold1:
            point_auto_n_minus_3 = 1
            calc_point_auto += 1
        
        idx_plus_3 = median_idx + 3
        if idx_plus_3 in ratio_map and ratio_map[idx_plus_3] >= auto_threshold2:
            point_auto_n_plus_3 = 1
            calc_point_auto += 1

    flat_results['median_ratio'] = median_val
    flat_results['median_peak_idx'] = median_idx
    flat_results['calc_point_auto'] = calc_point_auto
    flat_results['point_auto_n_minus_3'] = point_auto_n_minus_3
    flat_results['point_auto_n_plus_3'] = point_auto_n_plus_3

    avg_val = np.nan
    center_idx = -1
    calc_point_avg = 0
    point_avg_n_minus = 0
    point_avg_n_plus = 0
    
    if use_average_center and all_valid_ratios:
        avg_val = np.mean(all_valid_ratios)
        if use_global_mode:
            if isinstance(global_center_peak_idx, (list, tuple)) and len(global_center_peak_idx) >= 2:
                n1 = global_center_peak_idx[0]
                n2 = global_center_peak_idx[1]
                if n1 > 0:
                     idx_minus = n1 - center_offset
                     if idx_minus in ratio_map and ratio_map[idx_minus] >= auto_threshold1:
                         point_avg_n_minus = 1
                         calc_point_avg += 1
                if n2 > 0:
                     idx_plus = n2 + center_offset
                     if idx_plus in ratio_map and ratio_map[idx_plus] >= auto_threshold2:
                         point_avg_n_plus = 1
                         calc_point_avg += 1
                center_idx = n1 
            else:
                center_idx = global_center_peak_idx
                if center_idx != -1:
                    idx_minus = center_idx - center_offset
                    if idx_minus in ratio_map and ratio_map[idx_minus] >= auto_threshold1:
                        point_avg_n_minus = 1
                        calc_point_avg += 1
                    idx_plus = center_idx + center_offset
                    if idx_plus in ratio_map and ratio_map[idx_plus] >= auto_threshold2:
                        point_avg_n_plus = 1
                        calc_point_avg += 1
        else:
            if manual_center_idx is not None and manual_center_idx > 0:
                center_idx = manual_center_idx
            else:
                best_diff_avg = float('inf')
                for n, v in ratio_map.items():
                    diff = abs(v - avg_val)
                    if diff < best_diff_avg:
                        best_diff_avg = diff
                        center_idx = n
            
            if center_idx != -1:
                idx_minus = center_idx - center_offset
                if idx_minus in ratio_map and ratio_map[idx_minus] >= auto_threshold1:
                    point_avg_n_minus = 1
                    calc_point_avg += 1
                idx_plus = center_idx + center_offset
                if idx_plus in ratio_map and ratio_map[idx_plus] >= auto_threshold2:
                    point_avg_n_plus = 1
                    calc_point_avg += 1

    flat_results['avg_ratio'] = avg_val
    flat_results['center_peak_idx'] = center_idx
    flat_results['calc_point_avg'] = calc_point_avg
    flat_results['point_avg_n_minus'] = point_avg_n_minus
    flat_results['point_avg_n_plus'] = point_avg_n_plus

    return pd.Series(flat_results)

def prepare_analysis_dataframe_page3(df, group_interval, top_n_peaks, ratios_to_calc, auto_threshold1=0.6, auto_threshold2=0.3,
                                     use_average_center=False, manual_center_idx=0, center_offset=3,
                                     use_global_mode=False, global_center_peak_idx=0):
    if df.empty: return pd.DataFrame()
    df["group"] = (df["time"] / group_interval).apply(math.ceil)
    analysis_df = df.groupby('group').apply(
        analyze_group_page3, 
        top_n_peaks=top_n_peaks, 
        ratios_to_calc=ratios_to_calc,
        auto_threshold1=auto_threshold1,
        auto_threshold2=auto_threshold2,
        use_average_center=use_average_center,
        manual_center_idx=manual_center_idx,
        center_offset=center_offset,
        use_global_mode=use_global_mode,
        global_center_peak_idx=global_center_peak_idx
    )
    if analysis_df.empty: return pd.DataFrame()
    analysis_df['new_time'] = analysis_df.index * group_interval
    return analysis_df.reset_index(drop=True)

def main_process_page3(file_path, group_interval, top_n_peaks, ratios_to_calc, output_folder, auto_threshold1=0.6, auto_threshold2=0.3,
                       use_average_center=False, manual_center_idx=0, center_offset=3,
                       use_global_mode=False, global_center_peak_idx=0, validation_min_th=20):
    try:
        df = load_csv_page3(file_path)
        max_peak_value = df['peak_value'].max() if not df.empty else 0
        analysis_df = prepare_analysis_dataframe_page3(df, group_interval, top_n_peaks, ratios_to_calc, auto_threshold1, auto_threshold2,
                                                       use_average_center, manual_center_idx, center_offset,
                                                       use_global_mode, global_center_peak_idx)
        if not analysis_df.empty:
            os.makedirs(output_folder, exist_ok=True)
            base_filename = os.path.basename(file_path)
            output_filename = os.path.splitext(base_filename)[0] + "_peakcounts_peakvalue_ratio.csv"
            output_path = os.path.join(output_folder, output_filename)
            analysis_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            avg_ratio_8_1 = np.nan
            avg_ratio_13_1 = np.nan
            if 'peak_8/peak_1' in analysis_df.columns:
                avg_ratio_8_1 = analysis_df['peak_8/peak_1'].mean()
            if 'peak_13/peak_1' in analysis_df.columns:
                avg_ratio_13_1 = analysis_df['peak_13/peak_1'].mean()
            
            # Validation
            is_valid, sigma3, split_inv = validate_peak_data(df['peak_value'], min_th=validation_min_th)

            return {
                "file_name": base_filename,
                "avg_ratio_8_1": avg_ratio_8_1,
                "avg_ratio_13_1": avg_ratio_13_1,
                "max_peak_value": max_peak_value,
                "validation_result": is_valid, # 0 or 1
                "sigma3": sigma3,
                "split_invalid_count": split_inv,
                "analysis_df": analysis_df
            }
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- ロジッククラス ---

class LogicPage1:
    def create_plot_data_p1(self, df, frequency_column, spectrum_column, file_name, interval_width, ratio_threshold, save_folder_name, max_target_peak_index, y_max=0, base_dir=None):
        print("-" * 50)
        print(f"ファイル '{os.path.basename(file_name)}' の解析を開始します。(ページ1・第3peakまで選択・メモリ線ver)")

        total_points = 0
        peak_results_list = []
        top_peaks_df = pd.DataFrame()
        comparison_peaks_df = pd.DataFrame()
        point_winning_peaks_df = pd.DataFrame()

        analysis_df = df[(df[frequency_column] >= 0) & (df[frequency_column] <= 12)].copy()

        if analysis_df.empty:
            print("警告: 0-12Hzの範囲にデータが見つかりませんでした。")
            return 0

        if interval_width <= 0:
            print("エラー: 区間幅は0より大きい値でなければなりません。")
            return 0
        
        # --- 全範囲で一度だけピークを検出 ---
        print("0-12Hzの全範囲でピークを検出します...")
        all_peaks_indices, _ = find_peaks(analysis_df[spectrum_column])
        if len(all_peaks_indices) == 0:
            print("警告: 0-12Hzの範囲にピークが検出されませんでした。")
            return 0
        
        all_peaks_df = analysis_df.iloc[all_peaks_indices].copy()
        print(f"合計 {len(all_peaks_df)} 個のピークが検出されました。")

        num_ranges = int(12 / interval_width)
        peak_search_ranges = []
        for i in range(num_ranges):
            start = i * interval_width
            end = (i + 1) * interval_width
            peak_search_ranges.append((start, end))

        for range_min, range_max in peak_search_ranges:
            range_str = f"{range_min:.1f}-{range_max:.1f}Hz"
            print(f"\n--- 範囲: {range_str} の解析 ---")
            
            peak_data = all_peaks_df[
                (all_peaks_df[frequency_column] >= range_min) & 
                (all_peaks_df[frequency_column] < range_max)
            ].copy()

            if not peak_data.empty:
                if 2 <= range_min and range_max <= 10:
                    if len(peak_data) < 2:
                        print("  この範囲のピークが2つ未満のため、比率計算をスキップします。")
                        continue
                    
                    peak_data = peak_data.sort_values(by=spectrum_column, ascending=False)
                    top_peak = peak_data.iloc[0]
                    second_peak = peak_data.iloc[1]
                    
                    top_peaks_df = pd.concat([top_peaks_df, top_peak.to_frame().T])
                    
                    ratio1_2 = 0
                    if top_peak[spectrum_column] > 0:
                        ratio1_2 = second_peak[spectrum_column] / top_peak[spectrum_column]
                    
                    condition1_met = ratio1_2 >= ratio_threshold
                    
                    if max_target_peak_index == 2:
                        comparison_peaks_df = pd.concat([comparison_peaks_df, second_peak.to_frame().T])
                        points_added = 1 if condition1_met else 0
                        total_points += points_added
                        
                        print(f"  -> 1番目と2番目のピークを比較: 比率: {ratio1_2:.2%}")
                        if condition1_met:
                            print(f"     ★★★ 比率が{ratio_threshold:.0%}以上です！ (1ポイント獲得) ★★★")
                            point_winning_peaks_df = pd.concat([point_winning_peaks_df, second_peak.to_frame().T])

                        peak_results_list.append({
                            '範囲': range_str,
                            '1番目のピーク周波数(Hz)': f"{top_peak[frequency_column]:.2f}",
                            '1番目のピーク値': f"{top_peak[spectrum_column]:.2f}",
                            '比較対象ピーク': '第2ピーク',
                            '比較ピーク周波数(Hz)': f"{second_peak[frequency_column]:.2f}",
                            '比較ピーク値': f"{second_peak[spectrum_column]:.2f}",
                            '比率': f"{ratio1_2:.2%}",
                            'ポイント獲得': points_added
                        })

                    elif max_target_peak_index == 3:
                        if len(peak_data) < 3:
                            print("  この範囲のピークが3つ未満のため、第3ピークの特別判別をスキップします。")
                            continue
                        
                        third_peak = peak_data.iloc[2]
                        comparison_peaks_df = pd.concat([comparison_peaks_df, second_peak.to_frame().T, third_peak.to_frame().T])
                        
                        points_added_step1 = 0
                        if condition1_met:
                            points_added_step1 = 1
                            total_points += 1
                            point_winning_peaks_df = pd.concat([point_winning_peaks_df, second_peak.to_frame().T])

                        ratio2_3 = 0
                        points_added_step2 = 0
                        if condition1_met:
                            if second_peak[spectrum_column] > 0:
                                ratio2_3 = third_peak[spectrum_column] / second_peak[spectrum_column]
                            
                            condition2_met = ratio2_3 >= ratio_threshold
                            if condition2_met:
                                points_added_step2 = 1
                                total_points += 1
                                point_winning_peaks_df = pd.concat([point_winning_peaks_df, third_peak.to_frame().T])
                        
                        peak_results_list.append({
                            '範囲': range_str,
                            '1番目のピーク値': f"{top_peak[spectrum_column]:.2f}",
                            '2番目のピーク値': f"{second_peak[spectrum_column]:.2f}",
                            '3番目のピーク値': f"{third_peak[spectrum_column]:.2f}",
                            '比率(2vs1)': f"{ratio1_2:.2%}",
                            '比率(3vs2)': f"{ratio2_3:.2%}" if condition1_met else "N/A",
                            'ポイント獲得': points_added_step1 + points_added_step2
                        })
                else:
                    print("  この範囲は解析対象外のため、ポイント計算をスキップします。")
            else:
                print("  この範囲にピークはありませんでした。")

        # --- プロットデータの構築 ---
        plot_data = {
            'freq': analysis_df[frequency_column].tolist(),
            'fw': analysis_df[spectrum_column].tolist(),
            'all_peaks_freq': all_peaks_df[frequency_column].tolist() if not all_peaks_df.empty else [],
            'all_peaks_fw': all_peaks_df[spectrum_column].tolist() if not all_peaks_df.empty else [],
            'top_peaks_freq': top_peaks_df[frequency_column].tolist() if not top_peaks_df.empty else [],
            'top_peaks_fw': top_peaks_df[spectrum_column].tolist() if not top_peaks_df.empty else [],
            'comparison_peaks_freq': comparison_peaks_df[frequency_column].tolist() if not comparison_peaks_df.empty else [],
            'comparison_peaks_fw': comparison_peaks_df[spectrum_column].tolist() if not comparison_peaks_df.empty else [],
            'point_winning_peaks_freq': point_winning_peaks_df[frequency_column].tolist() if not point_winning_peaks_df.empty else [],
            'point_winning_peaks_fw': point_winning_peaks_df[spectrum_column].tolist() if not point_winning_peaks_df.empty else [],
            'title': f"周波数スペクトル (0-12Hz) - {os.path.basename(file_name)}\n合計ポイント: {total_points}"
        }

        # --- CSV保存ロジック (解析結果) ---
        target_base = base_dir if base_dir else os.path.dirname(os.path.abspath(sys.argv[0]))
        csv_dir = os.path.join(target_base, save_folder_name)
        os.makedirs(csv_dir, exist_ok=True)
        base_file_name = os.path.splitext(os.path.basename(file_name))[0]
        csv_path = os.path.join(csv_dir, f"analysis_result_{base_file_name}.csv")
        if peak_results_list:
            pd.DataFrame(peak_results_list).to_csv(csv_path, index=False, encoding='utf-8-sig')

        # --- グラフ描画 ---
        fig, ax = plt.subplots(figsize=(12, 7))
        if interval_width and interval_width > 0:
            num_ranges = int(12 / interval_width)
            for i in range(1, num_ranges):
                freq = i * interval_width
                ax.axvline(x=freq, color='gray', linestyle='--', linewidth=1, alpha=0.8)

        ax.plot(plot_data['freq'], plot_data['fw'], linestyle='-', color='royalblue', alpha=0.7, label="スペクトルデータ")
        ax.plot(plot_data['all_peaks_freq'], plot_data['all_peaks_fw'], 'o', color='gray', markersize=5, label="全検出ピーク", linestyle='None')
        ax.plot(plot_data['comparison_peaks_freq'], plot_data['comparison_peaks_fw'], 'o', color='green', markersize=8, label="比較対象ピーク", linestyle='None')
        ax.plot(plot_data['top_peaks_freq'], plot_data['top_peaks_fw'], 'o', color='red', markersize=8, label="第1ピーク", linestyle='None')
        ax.plot(plot_data['point_winning_peaks_freq'], plot_data['point_winning_peaks_fw'], 'x', color='red', markersize=12, markeredgewidth=2, label="ポイント獲得", linestyle='None')
        
        ax.set_title(plot_data['title'], fontsize=16)
        ax.set_xlabel("周波数 (Hz)", fontsize=12)
        ax.set_ylabel("スペクトル値", fontsize=12)
        ax.grid(True)
        ax.set_xlim(0, 12)
        if y_max > 0:
            ax.set_ylim(0, y_max)
        ax.legend()
        
        img_dir = os.path.join(csv_dir, "graphs")
        os.makedirs(img_dir, exist_ok=True)
        img_filename = f"graph_p1_{base_file_name}.jpeg"
        img_path = os.path.join(img_dir, img_filename)
        try:
            fig.savefig(img_path, format='jpeg', dpi=200, bbox_inches='tight')
        except Exception as e:
            print(f"グラフ保存エラー: {e}")
        finally:
            plt.close(fig)
        
        return total_points

    def run_analysis(self, file_paths, base_folder_name, prefix_name, interval_width, ratio_value, max_target_peak_index, split_idx_start, split_idx_end, base_output_dir=None):
        setup_japanese_font()
        ratio_threshold = ratio_value / 100.0
        interval_str = str(interval_width).replace('.', '_')
        ratio_str = str(int(ratio_value))
        peak_str = f"peak{max_target_peak_index}"
        
        save_folder_name_base = f"{base_folder_name}_range{interval_str}_ratio{ratio_str}_{peak_str}"
        save_folder_name = f"{prefix_name}_{save_folder_name_base}" if prefix_name else save_folder_name_base
        
        summary_results = []
        for file_name in file_paths:
            try:
                extracted_id = extract_id(file_name, split_idx_start, split_idx_end)
                df = pd.read_csv(file_name)
                total_points = self.create_plot_data_p1(df, "freq_hz", "amplitude", file_name, interval_width, ratio_threshold, save_folder_name, max_target_peak_index, base_dir=base_output_dir)
                summary_results.append({
                    'ID': extracted_id, 
                    '合計ポイント': total_points,
                    'ファイル名': os.path.basename(file_name) 
                })
            except Exception as e:
                print(f"Error in P1: {e}")
        
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            if 'ID' in summary_df.columns:
                 summary_df = summary_df[['ID', '合計ポイント', 'ファイル名']]
            
            summary_filename = f'{save_folder_name}_summary_total_points.csv'
            target_base = base_output_dir if base_output_dir else os.path.dirname(os.path.abspath(sys.argv[0]))
            summary_csv_path = os.path.join(target_base, save_folder_name, summary_filename)
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            return summary_df
        return None

class LogicPage2:
    def create_plot_data_p2(self, df, frequency_column, spectrum_column, file_name, interval_width_peak, interval_width_sum, save_folder_name, y_max=0, base_dir=None):
        print("-" * 50)
        print(f"ファイル '{os.path.basename(file_name)}' の解析を開始します。(ページ2・凸形状・カスタム区間ver)")

        analysis_df = df[(df[frequency_column] >= 0) & (df[frequency_column] <= 12)].copy()
        if analysis_df.empty: return 0, 0, 0
        all_peaks_indices, _ = find_peaks(analysis_df[spectrum_column])
        if len(all_peaks_indices) == 0: return 0, 0, 0
        all_peaks_df = analysis_df.iloc[all_peaks_indices].copy()

        # --- 第1ピーク解析 ---
        num_ranges_peak = int((10 - 2) / interval_width_peak)
        peak_search_ranges = [(2 + i * interval_width_peak, 2 + (i + 1) * interval_width_peak) for i in range(num_ranges_peak)]
        top_peaks_list = []
        for range_min, range_max in peak_search_ranges:
            peak_data_in_range = all_peaks_df[(all_peaks_df[frequency_column] >= range_min) & (all_peaks_df[frequency_column] < range_max)]
            if not peak_data_in_range.empty:
                top_peak = peak_data_in_range.loc[peak_data_in_range[spectrum_column].idxmax()]
                top_peaks_list.append(top_peak)
        
        # --- ピーク和解析 ---
        num_ranges_sum = int((10 - 2) / interval_width_sum)
        sum_search_ranges = [(2 + i * interval_width_sum, 2 + (i + 1) * interval_width_sum) for i in range(num_ranges_sum)]
        peak_sums_list = []
        for range_min, range_max in sum_search_ranges:
            peak_data_in_range = all_peaks_df[(all_peaks_df[frequency_column] >= range_min) & (all_peaks_df[frequency_column] < range_max)]
            range_center_freq = (range_min + range_max) / 2
            if not peak_data_in_range.empty:
                peak_sum = peak_data_in_range[spectrum_column].sum()
                peak_sums_list.append({'freq': range_center_freq, 'sum': peak_sum})
            else:
                peak_sums_list.append({'freq': range_center_freq, 'sum': 0})

        # --- 「上に凸」判定とポイント計算 ---
        point_peak_convex = 0
        is_peak_convex = False
        if len(top_peaks_list) >= 3:
            top_peaks_df_for_eval = pd.DataFrame(top_peaks_list).sort_values(by=frequency_column)
            is_peak_convex = is_convex_upwards(top_peaks_df_for_eval[spectrum_column].tolist())
            if is_peak_convex: point_peak_convex = 1

        point_sum_convex = 0
        is_sum_convex = False
        if len(peak_sums_list) >= 3:
            peak_sums_df_for_eval = pd.DataFrame(peak_sums_list)
            is_sum_convex = is_convex_upwards(peak_sums_df_for_eval['sum'].tolist())
            if is_sum_convex: point_sum_convex = 1

        total_points = point_peak_convex + point_sum_convex
        
        top_peaks_df = pd.DataFrame(top_peaks_list) if top_peaks_list else pd.DataFrame()
        peak_sums_df = pd.DataFrame(peak_sums_list) if peak_sums_list else pd.DataFrame()

        # --- プロットデータの構築 ---
        plot_data = {
            'freq': analysis_df[frequency_column].tolist(),
            'fw': analysis_df[spectrum_column].tolist(),
            'all_peaks_freq': all_peaks_df[frequency_column].tolist(),
            'all_peaks_fw': all_peaks_df[spectrum_column].tolist(),
            'top_peaks_freq': top_peaks_df[frequency_column].tolist() if not top_peaks_df.empty else [],
            'top_peaks_fw': top_peaks_df[spectrum_column].tolist() if not top_peaks_df.empty else [],
            'peak_sums_freq': peak_sums_df['freq'].tolist() if not peak_sums_df.empty else [],
            'peak_sums_val': peak_sums_df['sum'].tolist() if not peak_sums_df.empty else [],
            'is_peak_convex': is_peak_convex,
            'is_sum_convex': is_sum_convex,
            'title': f"周波数スペクトル (0-12Hz) - {os.path.basename(file_name)}\n合計ポイント: {total_points}",
        }

        # --- グラフ描画 ---
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        
        interval_for_grid = min(interval_width_peak, interval_width_sum)
        if interval_for_grid and interval_for_grid > 0:
            for i in np.arange(2, 10 + interval_for_grid, interval_for_grid):
                 ax.axvline(x=i, color='gray', linestyle='--', linewidth=1, alpha=0.8)

        ax.plot(plot_data['freq'], plot_data['fw'], linestyle='-', color='royalblue', alpha=0.7, label="スペクトルデータ")
        ax.plot(plot_data['all_peaks_freq'], plot_data['all_peaks_fw'], 'o', color='gray', markersize=5, label="全検出ピーク", linestyle='None')
        
        if plot_data['top_peaks_freq']:
            points = sorted(zip(plot_data['top_peaks_freq'], plot_data['top_peaks_fw']))
            sorted_freq, sorted_fw = zip(*points)
            line_color = 'red' if plot_data['is_peak_convex'] else 'green'
            label_text = f"第1ピーク ({'上に凸' if plot_data['is_peak_convex'] else '凸でない'})"
            ax.plot(sorted_freq, sorted_fw, marker='o', linestyle='-', color=line_color, 
                    markerfacecolor='red', markersize=8, label=label_text)

        ax.set_title(plot_data['title'], fontsize=16)
        ax.set_xlabel("周波数 (Hz)", fontsize=12)
        ax.set_ylabel("スペクトル値", fontsize=12, color='black')
        ax.tick_params(axis='y', labelcolor='black', direction='in')
        ax.tick_params(axis='x', direction='in')
        ax.grid(True, axis='x')
        ax.set_xlim(0, 12)
        
        if y_max > 0:
            ax.set_ylim(0, y_max)
        
        if plot_data['peak_sums_freq']:
            line_color_sum = 'hotpink' if plot_data['is_sum_convex'] else 'yellowgreen'
            label_text_sum = f"ピーク和 ({'上に凸' if plot_data['is_sum_convex'] else '凸でない'})"
            ax2.plot(plot_data['peak_sums_freq'], plot_data['peak_sums_val'],
                     marker='D', linestyle='--', color=line_color_sum,
                     markerfacecolor='hotpink', markersize=7, label=label_text_sum)

        ax2.set_ylabel("区間ごとのピーク値の合計", fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black', direction='in')
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        
        target_base = base_dir if base_dir else os.path.dirname(os.path.abspath(sys.argv[0]))
        img_dir = os.path.join(target_base, save_folder_name, "graphs")
        os.makedirs(img_dir, exist_ok=True)
        img_filename = f"graph_p2_{os.path.splitext(os.path.basename(file_name))[0]}.jpeg"
        img_path = os.path.join(img_dir, img_filename)
        try:
            fig.savefig(img_path, format='jpeg', dpi=200, bbox_inches='tight')
        except Exception as e:
            print(f"グラフ保存エラー: {e}")
        finally:
            plt.close(fig) 
        
        return total_points, point_peak_convex, point_sum_convex


    def run_analysis(self, file_paths, base_folder_name, prefix_name, interval_peak, interval_sum, split_idx_start, split_idx_end, base_output_dir=None):
        setup_japanese_font()
        interval_peak_str = str(interval_peak).replace('.', '_')
        interval_sum_str = str(interval_sum).replace('.', '_')
        save_folder_name_base = f"{base_folder_name}_peakRange{interval_peak_str}_sumRange{interval_sum_str}"
        save_folder_name = f"{prefix_name}_{save_folder_name_base}" if prefix_name else save_folder_name_base
        
        summary_results = []
        for file_name in file_paths:
            try:
                extracted_id = extract_id(file_name, split_idx_start, split_idx_end)
                df = pd.read_csv(file_name)
                total_points, p1, p2 = self.create_plot_data_p2(df, "freq_hz", "amplitude", file_name, interval_peak, interval_sum, save_folder_name, base_dir=base_output_dir)
                summary_results.append({
                    'ID': extracted_id, 
                    '合計ポイント': total_points,
                    '第1ピーク凸ポイント': p1,
                    'ピーク和凸ポイント': p2,
                    'ファイル名': os.path.basename(file_name) 
                })
            except Exception as e:
                print(f"Error in P2: {e}")
        
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            if 'ID' in summary_df.columns:
                 summary_df = summary_df[['ID', '合計ポイント', '第1ピーク凸ポイント', 'ピーク和凸ポイント', 'ファイル名']]
            
            target_base = base_output_dir if base_output_dir else os.path.dirname(os.path.abspath(sys.argv[0]))
            summary_dir = os.path.join(target_base, save_folder_name)
            os.makedirs(summary_dir, exist_ok=True)
            summary_filename = f'{save_folder_name}_total_points.csv'
            summary_csv_path = os.path.join(summary_dir, summary_filename)
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            
            return summary_df
        return None

class LogicPage3:
    def scan_for_global_mode(self, file_paths, group_interval, top_n_peaks, target_val):
        """パス1: 全ファイルをスキャンしてグローバルモードN（ターゲット比率に最も近いピークインデックス）を見つける"""
        print(f"--- グローバルモードスキャン (ターゲット: {target_val}) ---")
        all_closest_indices = []
        target_nums = range(2, 14)
        ratios_req = [(n, 1) for n in target_nums]

        for file_path in file_paths:
            try:
                df = load_csv_page3(file_path)
                if df.empty: continue
                df["group"] = (df["time"] / group_interval).apply(math.ceil)
                
                def get_closest_in_group(g_df):
                    res = cal_peakvalue_ratio_page3(g_df['peak_value'], 13, ratios_req)
                    best_n = -1
                    min_diff = float('inf')
                    for n in target_nums:
                        key = f'ratio_{n}_1'
                        val = res.get(key, np.nan)
                        if not pd.isna(val):
                            diff = abs(val - target_val)
                            if diff < min_diff:
                                min_diff = diff
                                best_n = n
                    return best_n

                group_results = df.groupby('group').apply(get_closest_in_group)
                valid_indices = group_results[group_results != -1].tolist()
                all_closest_indices.extend(valid_indices)
            except Exception as e:
                print(f"Scan Error ({file_path}): {e}")
        
        if not all_closest_indices:
            return 0
            
        try:
            mode_n = statistics.mode(all_closest_indices)
            return mode_n
        except statistics.StatisticsError:
            from collections import Counter
            c = Counter(all_closest_indices)
            most_common = c.most_common(1)
            if most_common: return most_common[0][0]
            return 0

    def run_analysis_mode(self, file_paths, base_folder_name, prefix_name, mode_config, base_output_dir=None):
        # mode_config: 次のキーを持つ辞書: mode_name, top_n, ratio1, ratio2, th1, th2, group_time, offset, center, exclusion_th
        mode = mode_config.get('mode_name', "1ファイルにつき1回判定")
        group_interval = float(mode_config.get('group_time', 2.0))
        top_n = int(mode_config.get('top_n', 13))
        r1_str = mode_config.get('ratio1', "8/1")
        r2_str = mode_config.get('ratio2', "13/1")
        th1 = float(mode_config.get('th1', 0.4))
        th2 = float(mode_config.get('th2', 0.2))
        exclusion_th = float(mode_config.get('exclusion_th', 20.0))
        split_start = int(mode_config.get('split_start', 2))
        split_end = int(mode_config.get('split_end', 4))
        
        ratios_to_calc = []
        def parse(s):
            try: return tuple(map(int, s.split('/')))
            except: return None
        
        if "自動" not in mode:
            rp1 = parse(r1_str)
            if rp1: ratios_to_calc.append(rp1)
            rp2 = parse(r2_str)
            if rp2: ratios_to_calc.append(rp2)

        # 平均/グローバルロジック
        use_avg = "平均値基準" in mode or "全体最頻値基準" in mode
        is_global = "全体最頻値基準" in mode
        center_offset = int(mode_config.get('offset', 3))
        manual_center = int(mode_config.get('center', 0))
        
        global_n_t1 = 0
        global_n_t2 = 0
        
        if is_global:
            global_n_t1 = self.scan_for_global_mode(file_paths, group_interval, top_n, th1)
            global_n_t2 = self.scan_for_global_mode(file_paths, group_interval, top_n, th2)
            print(f">>> Global N: T1({th1})->{global_n_t1}, T2({th2})->{global_n_t2}")

        settings_suffix = f"time{str(group_interval).replace('.','p')}_" + mode.replace("(", "_").replace(")", "")
        final_folder = f"{prefix_name}_{settings_suffix}" if prefix_name else settings_suffix
        script_dir = base_output_dir if base_output_dir else os.path.dirname(os.path.abspath(sys.argv[0]))
        output_folder = os.path.join(script_dir, "時間区間比率結果", final_folder)
        
        summary_results = []
        for file_path in file_paths:
            file_summary = main_process_page3(file_path, group_interval, top_n, ratios_to_calc, output_folder, th1, th2,
                                              use_average_center=use_avg, manual_center_idx=manual_center, center_offset=center_offset,
                                              use_global_mode=is_global, global_center_peak_idx=(global_n_t1, global_n_t2) if is_global else 0,
                                              validation_min_th=exclusion_th)
            

            
            if file_summary:
                extracted_id = extract_id(file_summary['file_name'], split_start, split_end)
                # max_val = file_summary.get("max_peak_value", 0)
                # valid = 1 if max_val >= exclusion_th else 0
                valid = file_summary.get("validation_result", 0) # Use the validation logic result
                
                analysis_df = file_summary.get("analysis_df")
                points = 0
                p1 = 0
                p2 = 0
                
                if analysis_df is not None:
                    row_points = []
                    row_p_r1 = []
                    row_p_r2 = []
                    
                    for _, row in analysis_df.iterrows():
                        rp = 0
                        rp1 = 0
                        rp2 = 0
                        
                        if use_avg:
                            rp = row.get('calc_point_avg', 0)
                            rp1 = row.get('point_avg_n_minus', 0)
                            rp2 = row.get('point_avg_n_plus', 0)
                        elif "自動" in mode:
                            rp = row.get('calc_point_auto', 0)
                            rp1 = row.get('point_auto_n_minus_3', 0)
                            rp2 = row.get('point_auto_n_plus_3', 0)
                        else:
                            # 通常
                            if len(ratios_to_calc) > 0:
                                 c1 = f'peak_{ratios_to_calc[0][0]}/peak_{ratios_to_calc[0][1]}'
                                 if c1 in row and row[c1] >= th1: rp1 = 1
                            if len(ratios_to_calc) > 1:
                                 c2 = f'peak_{ratios_to_calc[1][0]}/peak_{ratios_to_calc[1][1]}'
                                 if c2 in row and row[c2] >= th2: rp2 = 1
                            
                            if mode == "1ファイルにつき1回判定":
                                pass # 後で処理する
                            else:
                                rp = rp1 + rp2
                        
                        row_points.append(rp)
                        row_p_r1.append(rp1)
                        row_p_r2.append(rp2)
                    
                    # 集計
                    if mode == "1ファイルにつき1回判定":
                        avg_r1 = file_summary.get("avg_ratio_8_1", np.nan)
                        avg_r2 = file_summary.get("avg_ratio_13_1", np.nan)
                        out_p1 = 1 if (not pd.isna(avg_r1) and avg_r1 >= th1) else 0
                        out_p2 = 1 if (not pd.isna(avg_r2) and avg_r2 >= th2) else 0
                        points = out_p1 + out_p2
                        p1 = out_p1
                        p2 = out_p2
                    elif "平均化" in mode:
                         if row_points:
                             points = int(statistics.mean(row_points) + 0.5)
                             p1 = sum(row_p_r1)
                             p2 = sum(row_p_r2)
                    elif "最頻値" in mode:
                         if row_points:
                             modes = statistics.multimode(row_points)
                             points = max(modes)
                             p1 = sum(row_p_r1)
                             p2 = sum(row_p_r2)
                    else:
                         points = sum(row_points)
                         p1 = sum(row_p_r1)
                         p2 = sum(row_p_r2)
                    
                    if valid == 0: points = 0

                    res = {
                        'ID': extracted_id, 
                        'Point': points,
                        'Point_Ratio_8_1': p1,
                        'Point_Ratio_13_1': p2,
                        '有効データ判定': valid, 
                        'ファイル名': file_summary['file_name'] 
                    }
                    if is_global:
                        mode_str = f"{global_n_t1}"
                        if global_n_t2 != global_n_t1 and global_n_t2 > 0:
                            mode_str += f", {global_n_t2}"
                        res['Most_Frequent_Mode_Num'] = mode_str
                    
                    summary_results.append(res)
        
        if summary_results:
             return pd.DataFrame(summary_results)
        return None

def create_summary_excel(p1_data, p2_data, p3_data, save_path):
    import pandas as pd
    import numpy as np

    current_df = pd.DataFrame(columns=['ID'])
    
    # 1. P1 (第2ピーク) をマージ
    if p1_data is not None and not p1_data.empty:
        p1_subset = p1_data[['ID', '合計ポイント']].copy()
        p1_subset.rename(columns={'合計ポイント': '第２ピークポイント'}, inplace=True)
        current_df = pd.merge(current_df, p1_subset, on='ID', how='outer')

    # 2. P3 (時間区間比率) をマージ
    if p3_data is not None and not p3_data.empty:
        cols_to_use = ['ID', 'Point', '有効データ判定']
        if 'Point_Ratio_8_1' in p3_data.columns: cols_to_use.append('Point_Ratio_8_1')
        if 'Point_Ratio_13_1' in p3_data.columns: cols_to_use.append('Point_Ratio_13_1')


        if '有効データ判定' not in p3_data.columns: p3_data['有効データ判定'] = 1
        
        p3_subset = p3_data[cols_to_use].copy()
        p3_subset.rename(columns={
            'Point': '時間区間比率のポイント'
        }, inplace=True)
        current_df = pd.merge(current_df, p3_subset, on='ID', how='outer')

    if not current_df.empty: current_df['ID'] = current_df['ID'].astype(str)
    
    final_df = current_df.copy()
    
    # 3. プレフィックス一致ロジックで P2 (山) をマージ
    if p2_data is not None and not p2_data.empty:
        base_ids = []
        if not final_df.empty and 'ID' in final_df.columns:
            base_ids = final_df['ID'].dropna().unique().tolist()
        
        mapped_p2_data = []
        for _, p2_row in p2_data.iterrows():
            p2_id = str(p2_row['ID']) 
            p2_points = p2_row['合計ポイント']
            found_match = False
            for base_id in base_ids:
                base_id_str = str(base_id) 
                # 完全一致またはプレフィックス一致を確認 (P2 ID は Base ID のプレフィックス)
                if base_id_str == p2_id or base_id_str.startswith(p2_id + "_"):
                    mapped_p2_data.append({'ID': base_id, '山ポイント_temp': p2_points})
                    found_match = True
            # match match_ids にない場合はそのまま保持
            if not found_match: mapped_p2_data.append({'ID': p2_id, '山ポイント_temp': p2_points})

        if mapped_p2_data:
            p2_mapped_df = pd.DataFrame(mapped_p2_data)
            # 複数の P2 ファイルが同じ Base ID にマップされる場合は最大値を取る（保守的？）または集計
            p2_agg_df = p2_mapped_df.groupby('ID')['山ポイント_temp'].max().reset_index()
            
            if final_df.empty:
                final_df = p2_agg_df.rename(columns={'山ポイント_temp': '山ポイント'})
            else:
                final_df = pd.merge(final_df, p2_agg_df, on='ID', how='outer')
                if '山ポイント' not in final_df.columns: final_df['山ポイント'] = np.nan
                final_df['山ポイント'] = final_df['山ポイント_temp'].combine_first(final_df['山ポイント'])
                final_df.drop(columns=['山ポイント_temp'], inplace=True, errors='ignore')

    if 'ID' in final_df.columns:
        final_df.dropna(subset=['ID'], inplace=True)
    
    if final_df.empty:
        final_df.to_excel(save_path, index=False)
        return final_df

    # 4. 合計計算
    cols_check = ['第２ピークポイント', '山ポイント', '時間区間比率のポイント', '有効データ判定']
    for col in cols_check:
        if col not in final_df.columns: final_df[col] = 0 # 欠損列のデフォルト値を0にしてNaN問題を回避
        else: final_df[col] = final_df[col].fillna(0)
    
    final_df['合計ポイント'] = final_df['第２ピークポイント'] + final_df['山ポイント'] + final_df['時間区間比率のポイント']
    
    # 5. 列の並び順
    final_columns = ['ID', '合計ポイント', '第２ピークポイント', '山ポイント', '時間区間比率のポイント', '有効データ判定']
    # オプション列が存在する場合は追加
    extras = ['Average_Median_Peak_Idx']
    for c in extras:
        if c in final_df.columns: final_columns.append(c)
    
    # 実際に存在する列のみを保持（メイン列の作成は強制したが）
    final_columns_exist = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[final_columns_exist].copy()
    
    # 6. IDの第2部分でソート
    try:
        def extract_sort_key(id_val):
            try:
                parts = str(id_val).split('_')
                if len(parts) >= 2:
                    return int(parts[1])
                return 999999
            except:
                return 999999

        final_df['_sort_key'] = final_df['ID'].apply(extract_sort_key)
        final_df.sort_values(by=['_sort_key', 'ID'], inplace=True)
        final_df.drop(columns=['_sort_key'], inplace=True)
    except Exception:
        pass
    
    final_df.to_excel(save_path, index=False, engine='openpyxl')
    return final_df

if __name__ == "__main__":
    pass
