import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
from datetime import datetime
import pytz
import tkinter as tk
from tkinter import filedialog, messagebox

# --- 設定 ---
# フォント設定 (graph_create_v12.ipynbより)
FONT_FAMILY = ['Yu Gothic', 'Meiryo', 'Hiragino Sans', 'TakaoPGothic', 'IPAexGothic', 'Noto Sans CJK JP']
name='0609_1900'
def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False

def get_id_from_filename(filename):
    # ファイル名からIDを抽出 
    # 目標: data_peak_pulse_marged_20250609_190000_NCSA1A23090202_301_1_merged.csv -> 301_1
    base = os.path.basename(filename)
    name_body = os.path.splitext(base)[0]
    
    prefix = "data_peak_pulse_marged_20250609_190000_NCSA1A23090202_"
    suffix = "_merged"
    
    if name_body.startswith(prefix) and name_body.endswith(suffix):
        return name_body[len(prefix):-len(suffix)]
    return name_body

def select_folder():
    """GUIでフォルダを選択する"""
    root = tk.Tk()
    root.withdraw() # メインウィンドウを非表示
    folder_path = filedialog.askdirectory(title="処理対象のCSVファイルが入っているフォルダを選択してください")
    root.destroy()
    return folder_path

def process_single_file(file_path):
    """
    1つのCSVファイルを処理し、結果情報を返す。
    グラフの作成と保存もここで行う。
    有効なデータでない場合やエラー時は None を返す。
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipping {os.path.basename(file_path)}: Error reading file - {e}")
        return None
    
    # 必要な列の存在確認
    if 'value' not in df.columns or 'timestamp' not in df.columns:
        # print(f"Skipping {os.path.basename(file_path)}: Missing required columns.")
        return None

    values = df['value']
    timestamps = df['timestamp']
    
    if len(values) == 0:
        return None

    # --- 3シグマ計算と判定 ---
    std_dev = values.std()
    sigma_3 = 3 * std_dev
    
    # 判定: 3σが20以下または300以上なら無効(0)、それ以外は有効(1)
    is_valid = 1
    if sigma_3 <= 20 or sigma_3 >= 300:
        is_valid = 0
    
    # --- 5分割判定 (追加処理) ---
    # データを5分割して、それぞれの部分で3シグマ判定を行う
    # 5/2以上 (つまり3つ以上) が無効判定なら、全体を無効とする
    split_invalid_count = 0
    chunks = np.array_split(values, 5)
    
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            # データがない場合は便宜上無効カウントとするか？
            # ここでは分割時の端数処理等の安全策としてスキップまたは特定の扱いにするが、
            # 通常np.array_split等分ならデータがあれば0にはなりにくい。
            continue
            
        chunk_std = chunk.std()
        chunk_sigma_3 = 3 * chunk_std
        
        # 判定条件はこれまでと同じ
        if chunk_sigma_3 <= 20 or chunk_sigma_3 >= 300:
            split_invalid_count += 1
            
    # 元々有効だった場合のみ、この条件で再判定を行い無効化する
    # (元々無効だったものは無効のまま)
    if is_valid == 1:
        if split_invalid_count >= 3:  # 5分の2.5以上 -> 3以上
            is_valid = 0
            print(f"  -> Re-evaluated as Invalid (Split Invalid Count: {split_invalid_count}/5)")

    id_name = get_id_from_filename(file_path)
    print(f"Processed: {id_name} | 3Sigma: {sigma_3:.2f} | Valid: {is_valid} | SplitInvalid: {split_invalid_count}/5")

    # --- グラフ作成 ---
    setup_plot_style()
    
    # 保存フォルダ設定 (カレントディレクトリ/graph/有効 or 無効)
    current_dir = os.getcwd()
    status_dir = f"{name}_有効" if is_valid == 1 else f"{name}_無効"
    output_dir = os.path.join(current_dir, "graph", status_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # タイムスタンプ変換
    jst = pytz.timezone('Asia/Tokyo')
    try:
        # timestampはミリ秒前提
        datetimes = [datetime.fromtimestamp(ts/1000, jst) for ts in timestamps]
    except Exception as e:
        print(f"Skipping graph for {id_name}: Timestamp conversion error - {e}")
        plt.close(fig)
        return None

    # プロット
    ax.plot(datetimes, values, color='#ff0000', label='value')
    
    # スタイル設定
    ax.set_xlabel('datetime')
    ax.set_ylabel('value', color='black')
    ax.tick_params(axis='y', labelcolor='black', colors='black', direction='in')
    ax.tick_params(axis='x', direction='in')
    
    # グリッド
    ax.grid(which='major', linestyle='--', linewidth='0.7', color='gray')
    ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=5))
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # タイトルと凡例
    ax.set_title(id_name)
    ax.legend(['value'], loc='upper center', bbox_to_anchor=(0.5, -0.15))
    
    # 画像保存
    save_path = os.path.join(output_dir, f"{id_name}.jpg")
    try:
        plt.savefig(save_path, format='jpg', bbox_inches='tight')
    except Exception as e:
        print(f"Error saving graph for {id_name}: {e}")
    finally:
        plt.close(fig)

    # 結果を返す
    return {
        'ID': id_name,
        'Valid': is_valid,
        '3Sigma': sigma_3,
        'SplitInvalidCount': split_invalid_count
    }

def main():
    print("フォルダ選択ダイアログを開きます...")
    target_folder = select_folder()
    
    if not target_folder:
        print("フォルダが選択されませんでした。終了します。")
        return

    print(f"選択フォルダ: {target_folder}")
    
    # フォルダ内のCSVファイルを検索
    csv_files = glob.glob(os.path.join(target_folder, "*.csv"))
    
    if not csv_files:
        print("指定されたフォルダにCSVファイルが見つかりませんでした。")
        return

    print(f"{len(csv_files)} 個のCSVファイルが見つかりました。処理を開始します...")

    all_results = []
    
    # 全ファイルをループ処理
    for file_path in csv_files:
        result = process_single_file(file_path)
        if result:
            all_results.append(result)
            
    # 結果をまとめてCSV出力
    if all_results:
        current_dir = os.getcwd()
        result_csv_path = os.path.join(current_dir, f"{name}_validation_result.csv")
        
        df_result = pd.DataFrame(all_results)
        
        # カラム順序指定
        cols = ['ID', 'Valid', '3Sigma', 'SplitInvalidCount']
        df_result = df_result[cols]
        
        # カラム名変更
        df_result = df_result.rename(columns={'SplitInvalidCount': '分割後無効数'})
        
        df_result.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        
        print("-" * 30)
        print("全処理完了！")
        print(f"処理件数: {len(all_results)}")
        print(f"結果CSV: {result_csv_path}")
        
        # GUIメッセージで完了通知
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("完了", f"処理が完了しました。\n処理件数: {len(all_results)}\n結果CSV: {result_csv_path}")
        root.destroy()
    else:
        print("処理対象となる有効なデータが見つかりませんでした。")
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("結果", "処理対象となる有効なデータが見つかりませんでした。")
        root.destroy()

if __name__ == "__main__":
    main()
