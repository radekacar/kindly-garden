import os
import yaml
import cdsapi
import traceback
import asyncio
from typing import List, Dict, Any
import json

with open('cds_api_args.json', 'r') as f:
    api_args_list = json.load(f)

def load_credentials(path: str = "Copernicus.txt") -> Dict[str, str]:
    with open(path, "r") as f:
        creds = yaml.safe_load(f)
    return creds

async def download_one(args, output_dir, creds, semaphore):
    flight_id = args.get("flight_id", "unknown")
    out_name = os.path.join(output_dir, f"CDS_data_{flight_id}.grib")
    DATASET = "reanalysis-era5-pressure-levels"
    VARIABLES = [
        "geopotential", "temperature", "u_component_of_wind",
        "v_component_of_wind", "vertical_velocity"
    ]
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "pressure_level": args['pressure_level'],
        "year": args['year'],
        "month": args['month'],
        "day": args['day'],
        "time": args['time'],
        "area": args['area'],
        "format": "grib"
    }
    async with semaphore:
        try:
            client = cdsapi.Client(url=creds['url'], key=creds['key'])
            await asyncio.to_thread(client.retrieve, DATASET, request, out_name)
            return {"flight_id": flight_id, "status": "ok", "path": out_name}
        except Exception:
            tb = traceback.format_exc()
            return {"flight_id": flight_id, "status": "error", "error": tb}

async def download_cds_data_async(api_args_list: List[Dict[str, Any]], output_dir: str, creds_path: str = "Copernicus.txt") -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    creds = load_credentials(creds_path)
    semaphore = asyncio.Semaphore(4)
    tasks = [
        download_one(args, output_dir, creds, semaphore)
        for args in api_args_list
    ]
    results = await asyncio.gather(*tasks)
    oks = sum(1 for r in results if r['status'] == 'ok')
    errs = sum(1 for r in results if r['status'] == 'error')
    print(f"Downloads finished: {oks} succeeded, {errs} failed.")
    return results

async def main():
    results = await download_cds_data_async(api_args_list, 'dataset/ERA5')


if __name__ == "__main__":
    asyncio.run(main())
