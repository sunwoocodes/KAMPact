import pandas as pd
df = pd.read_csv("data/press_data_normal_with_idle.csv")
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
df = df.sort_values("TimeStamp").reset_index(drop=True)
gap = df["TimeStamp"].diff().dt.total_seconds().dropna()
odd = gap[(gap < 0.09) | ((gap > 0.11) & (gap < 1.0))]
print(len(gap), (gap.sub(0.1).abs() <= 0.01).sum(), (gap >= 1.0).sum(), len(odd))
for i in odd.index:
    print(i, gap[i], df.loc[i-1:i, "TimeStamp"].tolist())