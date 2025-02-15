import wrds
import pandas as pd

# Connect to WRDS
try:
    conn = wrds.Connection()
except Exception as e:
    print(f"WRDS Connection failed: {e}")

# Pull CRSP monthly stock data
crsp_m = conn.raw_sql("""
    SELECT permno, mthcaldt, mthret, shrout, mthprc
    FROM crsp.msf_v2
    WHERE mthcaldt BETWEEN '01/01/1959' AND '12/31/2022'
""", date_cols=['mthcaldt'])  # Ensures `mthcaldt` is datetime

# Ensure `mthcaldt` is correctly formatted as datetime before saving
crsp_m["mthcaldt"] = pd.to_datetime(crsp_m["mthcaldt"])

# Save the CSV with correct data types
crsp_m.to_csv("C:/Users/conno/OneDrive/Ken French data/crsp_m.csv", index=False)

print("CRSP monthly data has been saved.")

# Debugging Step: Reload the data to confirm types
reloaded_df = pd.read_csv("C:/Users/conno/OneDrive/Ken French data/crsp_m.csv", parse_dates=["mthcaldt"])
print(reloaded_df.dtypes)  # Ensure mthcaldt is datetime64[ns]
