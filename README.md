# PRC Data Challenge 2025

This repository contains the complete solution developed by the **kindly-garden** team for a data science challenge focused on aircraft fuel-burn prediction and flight data processing.
## Team & Acknowledgments

This repository is the official entry of the kindly-garden team for the data challenge.

**Team members:**
- [Rade Kačar](https://ba.linkedin.com/in/rade-kacar-88688b7b) — University of Belgrade, Faculty of Transport and Traffic Engineering
- [Darko Ćulibrk](https://ba.linkedin.com/in/darko-%C4%87ulibrk-400408a1) — MTEL BiH

We thank the challenge organizers and the open-source community for providing the tools and datasets that made this solution possible.

## Overview

This repository contains a workflow for preparing aviation data for a fuel-burn prediction task.

At a high level, the project does the following:

1. Downloads raw competition datasets from object storage.
2. Downloads weather data from ERA5 / Copernicus and ASOS-style meteorological sources.
3. Extracts flight segments from flight trajectories.
4. Computes segment-level statistics such as altitude, speed, and distance.
5. Detects climb / descent / cruise phases.
6. Estimates aircraft mass from energy-balance equations.
7. Derives fuel-flow-related features.
8. Produces ranked / final prediction files and uploads them.

---

## Pipeline Overview

A pipeline through this repository looks like this:

1. **Download source data**
   - [`downloader.py`](./downloader.py) downloads dataset objects from object storage into the `dataset/` directory.

2. **Main data processing and modeling**
   - [`data_processing_&_prediction.ipynb`](./data_processing_%26_prediction.ipynb) is the central notebook for data preparation, feature engineering, model training, validation, and prediction generation. Most of the workflow, from raw data to final predictions, is orchestrated here.  
   - **This notebook calls all the `multiprocessor_*.py` scripts (`multiprocessor_a.py`, `multiprocessor_b.py`, `multiprocessor_c.py`, `multiprocessor_d.py`, and `multiprocessor_e.py`) to perform the main data processing and feature engineering steps.**
3. **Build segment-level features**
   - [`multiprocessor_a.py`](./multiprocessor_a.py) extracts geographic and altitude bounds per segment.
   - [`multiprocessor_b.py`](./multiprocessor_b.py) computes segment statistics and fills missing airspeed-related fields using weather data.

4. **Detect phases**
   - [`multiprocessor_c.py`](./multiprocessor_c.py) marks whether a segment contains climb/descent phases.
   - [`multiprocessor_d.py`](./multiprocessor_d.py) extracts and stores climb/descent/cruise phase trajectories.

5. **Estimate mass and fuel flow**
   - [`multiprocessor_e.py`](./multiprocessor_e.py) estimates aircraft mass from energy balance and computes fuel-flow features.

6. **Submit predictions**
   - [`uploader_rank.py`](./uploader_rank.py) uploads rank/evaluation submissions.
   - [`uploader_final.py`](./uploader_final.py) uploads the final submission artifact.
---

## Repository Structure

### Top-level files

| File / Folder | Role                                                                                                                                               |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`README.md`](./README.md) | Project documentation.                                                                                                                             |
| [`cds_api_args.json`](./cds_api_args.json) | Request payloads / argument definitions for Copernicus CDS ERA5 downloads.                                                                         |
| [`Copernicus.txt`](./Copernicus.txt) | Copernicus CDS credentials/configuration.                                                                                                          |
| [`credentials.json`](./credentials.json) | MinIO / object storage credentials used by `downloader.py`.                                                                                        |
| [`downloader.py`](./downloader.py) | Downloads dataset files from object storage into `dataset/` and checks parquet health.                                                             |
| [`multiprocessor_a.py`](./multiprocessor_a.py) | Multiprocessing job for segment-level min/max latitude, longitude, and altitude statistics; also logs missing files and timeranges.                |
| [`multiprocessor_b.py`](./multiprocessor_b.py) | Core feature-engineering module for TAS/groundspeed filling, distance calculations, and enriched segment statistics using ERA5 weather.            |
| [`multiprocessor_c.py`](./multiprocessor_c.py) | Detects whether segments contain climb / descent phases using `openap.phase.FlightPhase`.                                                          |
| [`multiprocessor_d.py`](./multiprocessor_d.py) | Extracts climb / descent / cruise sub-trajectories and stores them as parquet files in `dataset/phases/`-style folders.                            |
| [`multiprocessor_e.py`](./multiprocessor_e.py) | Physics-based estimation module: fills TAS, computes drag/thrust/energy rate, estimates aircraft mass, and derives fuel-flow features.             |
| [`data_processing_&_prediction.ipynb`](./data_processing_%26_prediction.ipynb) | Main notebook for data preparation, experimentation, feature engineering, model training, validation, and prediction generation.                   |
| [`uploader_rank.py`](./uploader_rank.py) | Uploads versioned rank/evaluation submissions to the team bucket and logs returned scores.                                                         |
| [`uploader_final.py`](./uploader_final.py) | Uploads the selected final prediction file and checks submission status.                                                                           |
| [`health_check.txt`](./health_check.txt) | Output log from `downloader.py` indicating whether downloaded parquet files were readable.                                                         |
| [`missing_columns.log`](./missing_columns.log) | Log of missing expected columns in trajectory parquet files.  **Generated by [`multiprocessor_a.py`](./multiprocessor_a.py)**.                     |
| [`missing_parquet_files.log`](./missing_parquet_files.log) | Log of trajectory parquet files that were expected but not found/readable.  **Generated by [`multiprocessor_a.py`](./multiprocessor_a.py)**.       |
| [`missing_timeranges.log`](./missing_timeranges.log) | Log of requested flight segment timeranges that had no matching trajectory rows.  **Generated by [`multiprocessor_a.py`](./multiprocessor_a.py)**. |
| [`dataset/`](./dataset/) | Main data lake for raw, enriched, intermediate, and submission-ready datasets.                                                                     |
| [`models/`](./models/) | Trained model artifacts, prediction parquet files, upload logs, and result JSONs.                                                                  |

---

### `dataset/`

The `dataset/` directory acts as the main storage area for both raw and derived data. It contains the following important assets:

| File / Folder | Role |
|---|---|
| `aircraft_data_exported.csv` | Exported / cleaned version of aircraft reference data. |
| `airports.csv` | Airport metadata reference table. |
| `apt.parquet` | Airport-related parquet dataset. |
| `apt_updated.parquet` | Updated airport parquet dataset. |
| `ERA5/` | ERA5 weather files downloaded for training / processing. |
| `ERA5_rank_final/` | ERA5 weather files or derived weather assets for rank/final stage processing. |
| `flights_train/` | Per-flight trajectory parquet files for training. |
| `flights_rank/` | Per-flight trajectory parquet files for rank/evaluation stage. |
| `flights_rank_final/` | Rank/final stage trajectory data. |
| `flights_final/` | Final-stage trajectory data. |
| `meteo_data/` | Daily meteorological parquet files downloaded by `metar.py`. |
| `phases/` | Saved climb / descent / cruise phase parquet files. |
| `phases_final/` | Final-stage phase parquet files. |
| `flightlist_train.parquet` | Train split list / manifest of flights. |
| `flightlist_rank.parquet` | Rank/evaluation split list / manifest of flights. |
| `flightlist_final.parquet` | Final-stage list / manifest of flights. |
| `fuel_train.csv` / `fuel_train.parquet` | Training labels / fuel-burn targets. |
| `flight_segment_stats.csv` / `flight_segment_stats.parquet` | Base segment statistics dataset. |
| `flight_segment_stats_final.parquet` | Final-stage segment statistics snapshot. |
| `flight_segment_stats_united.csv` / `flight_segment_stats_united.parquet` | Unified segment statistics dataset after merges / consolidation. |
| `flight_segment_stats_united_final.parquet` | Final-stage unified segment statistics. |
| `flight_segment_stats_united_with_phases.csv` / `.parquet` | Unified segment stats with phase flags / phase outputs attached. |
| `flight_segment_stats_united_with_phases_final.csv` / `.parquet` | Final-stage version of the same dataset. |
| `flight_segment_stats_united_with_estimated_mass.csv` / `.parquet` | Unified segment stats enriched with estimated mass and possibly fuel-flow features. |
| `flight_segment_stats_united_with_estimated_mass_final.parquet` | Final-stage mass-enriched dataset. |
| `final_flight_segments.parquet` | Selected / filtered flight segments ready for later stages. |
| `final_flight_segments_final.parquet` | Final-stage flight segments snapshot. |
| `final_flight_segments_with_meteo.csv` / `.parquet` | Flight segments enriched with weather data. |
| `final_flight_segments_with_meteo_final.parquet` | Final-stage weather-enriched segments. |

---

### `models/`

The `models/` directory stores trained models and submission-related artifacts.

| File | Role |
|---|---|
| `lgbm_fuel_burn_model_04_002.pkl` | Serialized LightGBM model. |
| `lgbm_fuel_burn_model_04_002_rank_prediction.parquet` | Rank/evaluation predictions produced by the model. |
| `lgbm_fuel_burn_model_04_002_final_prediction.parquet` | Final prediction file chosen for submission. |
| `kindly-garden_v60.parquet` | Versioned uploaded submission artifact. |
| `kindly-garden_v60.parquet_result.json` | Result/status JSON returned by the platform after upload. |
| `model_log.txt` | Experiment or model run log. |
| `uploaded_so_far_rank.txt` | Upload history log maintained by `uploader_rank.py`. |

---
