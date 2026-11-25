from motec_csv_to_df import load_motec_csv
from lap_analysis import lap_table, plot_laps

df = load_motec_csv("Data Examples/DylanLongRun.csv")
laps = lap_table(df)
print(laps)
plot_laps(laps)