import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import os
from pathlib import Path

import pandas as pd
from dask import dataframe as dd
import pyarrow as pa

data_path = Path(os.getcwd())/"data"

# Check Pandas <-> Dask datatype , operation
pdf = pd.read_csv(data_path/"split_aa")
pdf.dtypes
pdf.shape[0]
pdf["Errors?"].unique()
pdf["Use Chip"].unique()
pdf["Is Fraud?"].unique()

# Reading CSV convert date
# parse_dates, create problems, must skip
ddf1 = dd.read_csv(
    data_path/"split_aa",
    # parse_dates={"dt": [2, 3, 4, 5]},
    dtype={"Errors?": str}
)
# Visualize copute graph
chk_null = ddf1.isna().sum()
chk_null.visualize()
chk_null.compute()

uniq_u_chip = ddf1["Use Chip"].unique()
uniq_u_chip.visualize()
uniq_u_chip.compute()

ddf1["Errors?"].unique().compute()

# Incase of csv data have error rows, should 
# Specify column name & dtypes manually
# dtype 'category' also reduce amount of memory used
ddf_col = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]

ddf_dtypes = {
    'User':'int16', 
    'Card':'int16',
    'Year':'int16',
    'Month':'int16', 
    'Day':'int16',
    'Time':'string', 
    'Amount':'string',
    'Use Chip':'category',
    'Merchant Name':'string',
    'Merchant City':'string',
    'Merchant State':'string',
    'Zip':'float',
    'MCC':'int16',
    'Errors?':'category',
    'Is Fraud?':'string',
}

ddf1 = dd.read_csv(
    data_path/"split_aa",
    skiprows=1,
    names=ddf_col,
    dtype=ddf_dtypes,
)
uniq_u_chip = ddf1["Use Chip"].unique()
uniq_u_chip.visualize()
uniq_u_chip.compute()

ddf1.memory_usage(deep=True).compute()

# Enforce dtype for reading CSV


pyarrow_schema = pa.schema([
    pa.field('User', pa.int64()),
    pa.field('Card', pa.int64()),
    pa.field('Year', pa.int64()),
    pa.field('Month', pa.int64()),
    pa.field('Day', pa.int64()),
    pa.field('Time', pa.string()),
    pa.field('Amount', pa.string()),
    pa.field('Use Chip', pa.string()),
    pa.field('Merchant Name', pa.string()),
    pa.field('Merchant City', pa.string()),
    pa.field('Merchant State', pa.string()),
    pa.field('Zip', pa.float64()),
    pa.field('MCC', pa.int64()),
    pa.field('Errors?', pa.string()),
    pa.field('Is Fraud?', pa.string()),
])

# Test write parquet write append
parquet_path = data_path/"combined.parquet"

for i, ddf in enumerate([ddf1, ddf2, ddf3]):
    if parquet_path.exists():
        print("Parquet Schema")
        print(dd.read_parquet(parquet_path).dtypes)
        print(i)
        print(ddf.dtypes)
        ddf.to_parquet(parquet_path, append=True, schema=pyarrow_schema, write_index=False)
    else:
        print(i)
        print(ddf.dtypes)
        ddf.to_parquet(parquet_path, schema=pyarrow_schema, write_index=False)

ddf = dd.read_parquet(parquet_path)

ddf.isnull().sum().compute()
