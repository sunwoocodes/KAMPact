import pandas as pd

# 1. 원본 정상 데이터 로드
df_normal = pd.read_csv('data/press_data_normal.csv')

# 2. 시간 비교를 위해 TimeStamp를 datetime 형식으로 변환한 임시 컬럼 생성
df_normal['TimeStamp_dt'] = pd.to_datetime(df_normal['TimeStamp'])

# 3. 새로운 'Idle' 컬럼 생성 및 기본값 0(가동 중)으로 초기화
df_normal['Idle'] = 0

# 4. 유휴 구간 조건 설정 (앞서 파악한 실제 CSV 내 유휴 구간의 시작과 끝)
# 시작: 2022-07-12 00:59:53.992
# 종료: 2022-07-12 01:11:32.588
idle_mask = (df_normal['TimeStamp_dt'] >= '2022-07-12 00:59:53.992') & \
            (df_normal['TimeStamp_dt'] <= '2022-07-12 01:11:32.588')

# 5. 조건에 해당하는(유휴 구간인) 행의 'Idle' 값을 1로 변경
df_normal.loc[idle_mask, 'Idle'] = 1

# 6. 연산을 위해 만들었던 임시 datetime 컬럼 삭제 (원본 컬럼 구조 유지)
df_normal = df_normal.drop(columns=['TimeStamp_dt'])

# 7. 결과를 새로운 CSV 파일로 저장
output_filename = 'press_data_normal_with_idle.csv'
df_normal.to_csv(output_filename, index=False)

# 결과 확인 출력
print(f"파일이 성공적으로 저장되었습니다: {output_filename}")
print("\n[Idle 컬럼 데이터 분포]")
print(df_normal['Idle'].value_counts())