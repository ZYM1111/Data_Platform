import pandas as pd
import time

def f(df):
    df["123"] = df["tms"].apply(lambda x: x[0] if len(x) > 0 else None)
    return df

df = pd.read_csv("/home/users/lichuan/shared/jld2csv/trips_150103.jld2_trips.csv")

a = time.time()
df = f(df)
b = time.time()
print(b-a)
