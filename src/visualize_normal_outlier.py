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

def plot_styled_timeseries(df, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14)
    
    # 해당 데이터의 날짜 부분만 추출
    date_str = df['TimeStamp'].iloc[0].strftime('%Y-%m-%d')
    
    # 오른쪽 상단에 날짜 텍스트 표시
    fig.text(0.98, 0.96, f"Date: {date_str}", ha="right", va="bottom", fontsize=11, fontweight='bold', color='#333333')
    
    # 선 두께, 색상 및 격자 스타일 설정
    line_kwargs = {'linewidth': 0.5, 'color': '#1f77b4'}
    grid_kwargs = {'color': '#e0e0e0', 'linestyle': '-', 'linewidth': 0.5, 'alpha': 0.7}
    
    axes[0].plot(df['TimeStamp'], df['AI0_Vibration'], **line_kwargs)
    axes[0].set_ylabel('AI0_Vibration')
    axes[0].grid(True, **grid_kwargs)
    
    axes[1].plot(df['TimeStamp'], df['AI1_Vibration'], **line_kwargs)
    axes[1].set_ylabel('AI1_Vibration')
    axes[1].grid(True, **grid_kwargs)
    
    axes[2].plot(df['TimeStamp'], df['AI2_Current'], **line_kwargs)
    axes[2].set_ylabel('AI2_Current')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, **grid_kwargs)
    
    # [수정된 부분 1] 데이터 양 끝에 여백(전체 시간의 2%) 추가하여 x축 범위 설정
    time_span = df['TimeStamp'].max() - df['TimeStamp'].min()
    padding = time_span * 0.02
    
    start_time = df['TimeStamp'].min()
    end_time = df['TimeStamp'].max()
    
    axes[2].set_xlim(start_time - padding, end_time + padding)
    
    # 밀리초 3자리까지 출력하는 커스텀 포맷터 설정
    def time_fmt(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%H:%M:%S.%f')[:-3]
    
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(time_fmt))
    
    # [수정된 부분 2] 중간 눈금을 모두 지우고 시작 시간과 끝 시간만 x축 틱으로 설정
    start_num = mdates.date2num(start_time)
    end_num = mdates.date2num(end_time)
    
    axes[2].set_xticks([start_num, end_num])
    
    # x축 라벨 겹침 방지 (0도로 똑바로 표시)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=0)
    
    # 레이아웃 밀착 및 여백 조정
    plt.subplots_adjust(hspace=0.1) 
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    # 이미지 저장
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# 정상 데이터 시각화 실행
plot_styled_timeseries(df_normal, 'NORMAL Raw Sensor Time Series', 'normal_timeseries_with_padding.jpg')

# 이상 데이터 시각화 실행
plot_styled_timeseries(df_outlier, 'OUTLIER Raw Sensor Time Series', 'outlier_timeseries_with_padding.jpg')