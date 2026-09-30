import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# 1. 데이터 로드
df_normal = pd.read_csv('data/press_data_normal.csv')
df_outlier = pd.read_csv('data/outlier_data.csv')

# 2. TimeStamp 컬럼을 datetime 형식으로 변환 및 정렬
df_normal['TimeStamp'] = pd.to_datetime(df_normal['TimeStamp'])
df_normal = df_normal.sort_values('TimeStamp')

df_outlier['TimeStamp'] = pd.to_datetime(df_outlier['TimeStamp'])
df_outlier = df_outlier.sort_values('TimeStamp')

# 3. 그래프 생성 함수 정의 (is_normal 파라미터로 유휴구간 표시 여부 제어)
def plot_corrected_annotated_timeseries(df, title, filename, is_normal=False):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    
    # 우측 상단 날짜 표기
    date_str = df['TimeStamp'].iloc[0].strftime('%Y-%m-%d')
    fig.text(0.95, 0.96, f"Date: {date_str}", ha="right", va="bottom", fontsize=11, fontweight='bold', color='#333333')
    
    line_kwargs = {'linewidth': 0.5, 'color': '#1f77b4'}
    grid_kwargs = {'color': '#e0e0e0', 'linestyle': '-', 'linewidth': 0.5, 'alpha': 0.7}
    
    cols = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
    
    # 정상 데이터일 경우에만 수정된 유휴 구간(가장자리 스파이크 제외) 필터링
    if is_normal:
        idle_mask = (df['TimeStamp'] >= '2022-07-12 00:59:53.992') & (df['TimeStamp'] <= '2022-07-12 01:11:32.588')
        df_idle = df[idle_mask]

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(df['TimeStamp'], df[col], **line_kwargs)
        ax.set_ylabel(col)
        ax.grid(True, **grid_kwargs)
        
        # 전체 최대/최소값 계산
        overall_max = df[col].max()
        overall_min = df[col].min()
        
        # 텍스트 포맷 (진동: 소수점 3자리, 전류: 소수점 1자리)
        fmt = '{:.3f}' if 'Vibration' in col else '{:.1f}'
        
        # 전체 최대/최소 점선 그리기 (빨간색 얇은 파선)
        ax.axhline(overall_max, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(overall_min, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # 오른쪽에 전체 값 텍스트 표시
        trans = ax.get_yaxis_transform()
        ax.text(1.01, overall_max, f"Max: {fmt.format(overall_max)}", color='red', 
                transform=trans, va='center', ha='left', fontsize=9)
        ax.text(1.01, overall_min, f"Min: {fmt.format(overall_min)}", color='red', 
                transform=trans, va='center', ha='left', fontsize=9)
        
        # 정상 데이터의 경우에만 유휴 구간 최대/최소값 추가 (초록색 얇은 점선)
        if is_normal:
            idle_max = df_idle[col].max()
            idle_min = df_idle[col].min()
            
            ax.axhline(idle_max, color='green', linestyle=':', linewidth=1.5, alpha=0.8)
            ax.axhline(idle_min, color='green', linestyle=':', linewidth=1.5, alpha=0.8)
            
            # 텍스트가 겹치지 않도록 위/아래 정렬 조정
            ax.text(1.01, idle_max, f"Idle Max: {fmt.format(idle_max)}", color='green', 
                    transform=trans, va='bottom', ha='left', fontsize=9)
            ax.text(1.01, idle_min, f"Idle Min: {fmt.format(idle_min)}", color='green', 
                    transform=trans, va='top', ha='left', fontsize=9)

    axes[2].set_xlabel('Time')
    
    # x축 양 끝 여백 설정 (전체 시간의 2%)
    time_span = df['TimeStamp'].max() - df['TimeStamp'].min()
    padding = time_span * 0.02
    start_time = df['TimeStamp'].min()
    end_time = df['TimeStamp'].max()
    axes[2].set_xlim(start_time - padding, end_time + padding)
    
    # 시간 포맷터 설정 (밀리초 3자리 표기)
    def time_fmt(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%H:%M:%S.%f')[:-3]
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(time_fmt))
    
    # X축 틱 겹침 방지 필터링 (시작점, 끝점, 필터링된 중간점)
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)
    current_ticks = axes[2].get_xticks()
    tick_threshold = (end_num - start_num) * 0.10
    
    filtered_ticks = []
    for tick in current_ticks:
        if abs(tick - start_num) > tick_threshold and abs(tick - end_num) > tick_threshold:
            if start_num < tick < end_num:
                filtered_ticks.append(tick)
                
    final_ticks = [start_num] + filtered_ticks + [end_num]
    final_ticks.sort()
    axes[2].set_xticks(final_ticks)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0)
    
    # 레이아웃 조정 및 이미지 저장 (bbox_inches='tight' 필수: 우측 텍스트 잘림 방지)
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# 4. 함수 실행: 정상 데이터 (유휴 구간 표시 활성화)
plot_corrected_annotated_timeseries(
    df_normal, 
    title='NORMAL Raw Sensor Time Series (Corrected Idle)', 
    filename='normal_annotated_corrected.jpg', 
    is_normal=True
)

# 5. 함수 실행: 이상 데이터 (유휴 구간 표시 비활성화)
plot_corrected_annotated_timeseries(
    df_outlier, 
    title='OUTLIER Raw Sensor Time Series', 
    filename='outlier_annotated.jpg', 
    is_normal=False
)