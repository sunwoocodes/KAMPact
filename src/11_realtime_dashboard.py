import argparse
import platform
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# 한글 폰트 강제 탐색 및 설정 (메인 그래프 레이블용)
system_os = platform.system()
if system_os == 'Windows':
    target_font = 'Malgun Gothic'
elif system_os == 'Darwin':
    target_font = 'AppleGothic'
else:
    target_font = 'NanumGothic'

font_list = [f.name for f in fm.fontManager.ttflist]
if target_font in font_list:
    plt.rcParams['font.family'] = target_font
else:
    korean_fonts = [f for f in font_list if any(x in f for x in ['Gothic', 'Malgun', 'Nanum', 'Apple'])]
    if korean_fonts:
        plt.rcParams['font.family'] = korean_fonts[0]

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

try:
    from core_detector import StreamingDetector
except ImportError:
    from src.core_detector import StreamingDetector


def main():
    parser = argparse.ArgumentParser(description="KAMPact - OR_3_P2 Real-time Dashboard")
    parser.add_argument("--normal-path", default="data/press_data_normal_with_idle.csv")
    parser.add_argument("--fault-path", default="data/outlier_data.csv")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second")
    parser.add_argument("--plot-window", type=int, default=150, help="Number of samples to display")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. 데이터 로드 및 시연 준비 (전처리 포함)
    # ---------------------------------------------------------
    print("Loading data...")
    df_normal = pd.read_csv(args.normal_path)
    df_fault = pd.read_csv(args.fault_path)

    def preprocess(df, source_name):
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df.sort_values('TimeStamp', inplace=True)
        
        gap = df['TimeStamp'].diff().dt.total_seconds()
        brk = gap > 0.5
        brk.iloc[0] = True
        seg = brk.cumsum() - 1
        
        df['group_id'] = source_name + "_" + seg.astype(str)
        return df

    df_normal = preprocess(df_normal, "normal")
    df_fault = preprocess(df_fault, "fault")
    
    # ---------------------------------------------------------
    # 2. 코어 엔진 초기화 및 학습 (Threshold 고정)
    # ---------------------------------------------------------
    detector = StreamingDetector(
        win1_size=10, 
        win05_size=5, 
        sample_agg_k=3, 
        persistence_p=2, 
        quantile_threshold=0.9999
    )
    
    print("\nInitializing model and thresholds with normal data (fit)...")
    detector.fit(df_normal)
    
    demo_df = pd.concat([df_normal.tail(1000), df_fault]).reset_index(drop=True)
    total_frames = len(demo_df)
    print(f"\nReady to stream {total_frames} samples.")

    # ---------------------------------------------------------
    # 3. 플롯 버퍼 준비
    # ---------------------------------------------------------
    plot_window = args.plot_window
    x_data = np.arange(plot_window)
    
    buf_ai0 = deque([np.nan]*plot_window, maxlen=plot_window)
    buf_ai1 = deque([np.nan]*plot_window, maxlen=plot_window)
    buf_ai2 = deque([np.nan]*plot_window, maxlen=plot_window)
    buf_alarm = deque([0]*plot_window, maxlen=plot_window)

    # ---------------------------------------------------------
    # 4. GUI 플롯 레이아웃 설정
    # ---------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title('KAMPact Real-time Ensemble Dashboard (OR_3_P2)')
    
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.2], height_ratios=[2, 2, 1.2])
    
    ax_vib = fig.add_subplot(gs[0, 0])
    ax_cur = fig.add_subplot(gs[1, 0])
    ax_alm = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    # [1] 상단 그래프: 진동 (초기 Y축 범위는 임의로 설정하되, 실행 시 자동 조절됨)
    line_ai0, = ax_vib.plot(x_data, buf_ai0, label='AI0 (Vibration)', color='#00ffcc', alpha=0.8)
    line_ai1, = ax_vib.plot(x_data, buf_ai1, label='AI1 (Vibration)', color='#ff00ff', alpha=0.8)
    ax_vib.set_xlim(0, plot_window)
    ax_vib.set_title("Real-time Sensor Streaming (Vibration)", fontsize=12, fontweight='bold')
    ax_vib.legend(loc='upper right')
    ax_vib.grid(True, linestyle='--', alpha=0.3)

    # [2] 중단 그래프: 전류 
    line_ai2, = ax_cur.plot(x_data, buf_ai2, label='AI2 (Current)', color='#ffff00', alpha=0.8)
    ax_cur.set_xlim(0, plot_window)
    ax_cur.set_title("Real-time Sensor Streaming (Current)", fontsize=12, fontweight='bold')
    ax_cur.legend(loc='upper right')
    ax_cur.grid(True, linestyle='--', alpha=0.3)

    # [3] 하단 그래프: 최종 알람
    line_alarm, = ax_alm.plot(x_data, buf_alarm, color='red', linewidth=2)
    ax_alm.set_xlim(0, plot_window)
    ax_alm.set_ylim(-0.2, 1.2)
    ax_alm.set_yticks([0, 1])
    ax_alm.set_yticklabels(['Normal', 'FAULT!'], color='red', fontweight='bold')
    ax_alm.set_title("Final Ensemble Decision (OR_3_P2)", fontsize=12, fontweight='bold')

    # 우측 정보 텍스트 패널 (영문)
    ax_info.axis('off')
    info_text = ax_info.text(
        0.05, 0.95, "", 
        transform=ax_info.transAxes, 
        fontsize=12, 
        verticalalignment='top', 
        color='white', 
        family='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e1e1e", edgecolor="#555555", alpha=0.8)
    )

    plt.tight_layout()

    def get_status_tag(is_alarm):
        return "[ ALARM ]" if is_alarm else "[ NORMAL]"

    # 고정된 성능 지표 텍스트 (영문)
    PERF_TEXT = (
        "🏆 [VALIDATED MODEL PERFORMANCE]\n"
        " - Model      : OR_3_P2 (Ensemble)\n"
        " - Detection  : 91.43 %\n"
        " - False Alarm: 0.90 %\n"
        " - Avg Delay  : 0.57 sec\n\n"
        + "="*35 + "\n\n"
    )

    # ---------------------------------------------------------
    # 5. 애니메이션 업데이트 함수
    # ---------------------------------------------------------
    def update(frame):
        if frame >= total_frames:
            ani.event_source.stop()
            return line_ai0, line_ai1, line_ai2, line_alarm, info_text
            
        row = demo_df.iloc[frame]
        t = row['TimeStamp']
        ai0, ai1, ai2 = row['AI0_Vibration'], row['AI1_Vibration'], row['AI2_Current']

        res = detector.step(t, ai0, ai1, ai2)

        buf_ai0.append(ai0)
        buf_ai1.append(ai1)
        buf_ai2.append(ai2)
        buf_alarm.append(1 if res['final_alarm'] else 0)

        line_ai0.set_ydata(buf_ai0)
        line_ai1.set_ydata(buf_ai1)
        line_ai2.set_ydata(buf_ai2)
        line_alarm.set_ydata(buf_alarm)

        # 동적 Y축 스케일링 (자동 상하안 조절)
        # 진동 (AI0, AI1) 스케일링
        vib_data = np.concatenate([np.array(buf_ai0, dtype=float), np.array(buf_ai1, dtype=float)])
        if not np.all(np.isnan(vib_data)):
            vmin, vmax = np.nanmin(vib_data), np.nanmax(vib_data)
            margin = max((vmax - vmin) * 0.1, 0.1)
            ax_vib.set_ylim(vmin - margin, vmax + margin)

        # 전류 (AI2) 스케일링
        cur_data = np.array(buf_ai2, dtype=float)
        if not np.all(np.isnan(cur_data)):
            cmin, cmax = np.nanmin(cur_data), np.nanmax(cur_data)
            margin = max((cmax - cmin) * 0.1, 0.1)
            ax_cur.set_ylim(cmin - margin, cmax + margin)

        # 하단 알람 그래프 배경 업데이트
        for coll in list(ax_alm.collections):
            coll.remove()
        ax_alm.fill_between(x_data, 0, buf_alarm, color='red', alpha=0.3)

        # 우측 텍스트 패널 업데이트 (영문)
        thr_s = detector.thresholds['sample']
        thr_w05 = detector.thresholds['win05']
        thr_w1 = detector.thresholds['win1']
        time_str = t.strftime('%H:%M:%S.%f')[:-3]
        
        text = PERF_TEXT
        text += f"🕒 Time: {time_str}\n"
        text += "="*35 + "\n\n"
        text += "📊 [INDIVIDUAL DETECTOR SCORES]\n"
        text += f" Sample : {res['sample_score']:6.1f} / {thr_s:<5.1f} {get_status_tag(res['sample_alarm'])}\n"
        text += f" Win0.5 : {res['win05_score']:6.1f} / {thr_w05:<5.1f} {get_status_tag(res['win05_alarm'])}\n"
        text += f" Win1.0 : {res['win1_score']:6.1f} / {thr_w1:<5.1f} {get_status_tag(res['win1_alarm'])}\n\n"
        
        text += "="*35 + "\n\n"
        text += "⚙️ [ENSEMBLE LOGIC]\n"
        text += f" Early Warn (OR_3): {get_status_tag(res['or3_base'])}\n"
        history_sum = sum(detector.or3_history)
        text += f" P2 Persistence   : {history_sum} / {detector.p} (Held)\n\n"
        
        text += "="*35 + "\n\n"
        text += "🚨 [FINAL DECISION]\n"
        if res['final_alarm']:
            text += " >> FAULT DETECTED! <<\n"
            fig.patch.set_facecolor('#330000') 
        else:
            text += " >> NORMAL OPERATION\n"
            fig.patch.set_facecolor('black')
            
        info_text.set_text(text)

        return line_ai0, line_ai1, line_ai2, line_alarm, info_text

    # ---------------------------------------------------------
    # 6. 루프 실행
    # ---------------------------------------------------------
    interval_ms = int(1000 / args.fps)
    
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        interval=interval_ms, 
        blit=False
    )

    plt.show()

if __name__ == "__main__":
    main()