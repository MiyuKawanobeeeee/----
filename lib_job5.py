
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, os, re

# Initial config
plt.rcParams["figure.dpi"] = 100

def run_fft_analysis(target_folder_path, freq_range=(0, 10)):
    # target_folder_path: The folder containing .csv files to process
    
    print(f"FFT Analysis Target: {target_folder_path}")
    
    csv_out_folder = os.path.join(target_folder_path, "fft_csv1")
    graph_out_folder = os.path.join(csv_out_folder, "fft_graph1")
    os.makedirs(csv_out_folder, exist_ok=True)
    os.makedirs(graph_out_folder, exist_ok=True)
    
    # Target "HPF.csv"
    all_csv = glob.glob(os.path.join(target_folder_path, "*.csv"))
    csv_files = [f for f in all_csv if re.search(r'HPF\.csv$', os.path.basename(f), flags=re.IGNORECASE)]
    
    print(f"Found {len(csv_files)} files ending with HPF.csv.")
    
    if not csv_files:
        print("No match files found.")
        return

    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
            df.columns = df.columns.str.strip()

            if not {"timestamp", "value"}.issubset(df.columns):
                print(f"  Skip: {os.path.basename(file)} -> Columns missing")
                continue

            t_ms = pd.to_numeric(df["timestamp"], errors="coerce")
            x = pd.to_numeric(df["value"], errors="coerce")
            valid = (~t_ms.isna()) & (~x.isna())
            t_ms, x = t_ms[valid].values, x[valid].values

            if len(x) < 8:
                print(f"  Skip: Data insufficient -> {os.path.basename(file)}")
                continue

            order = np.argsort(t_ms)
            t_ms, x = t_ms[order], x[order]

            dt_ms = np.median(np.diff(t_ms))
            if dt_ms <= 0:
                print(f"  Skip: Invalid time interval -> {os.path.basename(file)}")
                continue
                
            dt = dt_ms / 1000.0
            fs = 1.0 / dt

            # Intepolation
            t0, t1 = t_ms[0]/1000.0, t_ms[-1]/1000.0
            n_uniform = int(np.floor((t1 - t0) / dt)) + 1
            t_uniform = t0 + np.arange(n_uniform) * dt
            x_uniform = np.interp(t_uniform, t_ms/1000.0, x)

            # FFT
            window = np.hanning(len(x_uniform))
            xw = x_uniform * window
            N = len(xw)
            X = np.fft.rfft(xw)
            freqs = np.fft.rfftfreq(N, d=dt)
            amp = np.abs(X) * (2.0 / np.sum(window))

            if freq_range is not None:
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                freqs_plot, amp_plot = freqs[mask], amp[mask]
            else:
                freqs_plot, amp_plot = freqs, amp

            base = os.path.splitext(os.path.basename(file))[0]

            csv_out = os.path.join(csv_out_folder, base + "_fft.csv")
            pd.DataFrame({"freq_hz": freqs, "amplitude": amp}).to_csv(csv_out, index=False, encoding="utf-8-sig")

            plt.figure(figsize=(12, 6))
            plt.plot(freqs_plot, amp_plot, linewidth=1.0)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude")
            if freq_range:
                plt.xlim(freq_range)
            plt.title(f"FFT | {base}\nFs={fs:.2f} Hz, N={N}, duration={t1-t0:.2f} s")
            plt.grid(True)
            plt.tight_layout()

            jpg_out = os.path.join(graph_out_folder, base + "_fft.jpg")
            plt.savefig(jpg_out, dpi=300)
            plt.close()

            print(f"  Processed: {os.path.basename(file)}")

        except Exception as e:
            print(f"  Error: {os.path.basename(file)} -> {e}")

if __name__ == "__main__":
    pass
