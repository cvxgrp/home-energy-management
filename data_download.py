import pandas as pd
import requests
from io import StringIO

def download_nordpool_data(year, area='NO3', currency='EUR'):
    url = f'https://www.nordpoolgroup.com/globalassets/marketdata-excel-files/elspot-prices_{year}_hourly_{currency}.zip'
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download the data for {year}. Status code: {response.status_code}")

    with open(f"elspot-prices_{year}_hourly_{currency}.zip", "wb") as f:
        f.write(response.content)

    df = pd.read_csv(f"elspot-prices_{year}_hourly_{currency}.zip", compression='zip', encoding='latin1', sep=';', header=0, decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.loc[df['Area'] == area].reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    year = 2022
    area = 'NO3'  # Trondheim is part of area NO3
    currency = 'NOK'

    df = download_nordpool_data(year, area, currency)
    df.to_csv(f"trondheim_electricity_prices_{year}.csv", index=False)

    print("Downloaded historical spot electricity prices for Trondheim from Nord Pool for the year 2020.")

