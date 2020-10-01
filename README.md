# GTV for Seasonal Forecasting

### Setup
1. Clone this repository to wherever you work (locally or on a server).
2. Create a new folder within the repository called `data`. Then, download the datasets from 
[here](https://drive.google.com/drive/folders/1qN2bp1_1bvH5KLOkoBrEEalDmAhIS-Xv?usp=sharing) and store them in the `data`
folder. Keeping this structure allows the code to run smoothly.

3. Install the Python dependencies in
[requirements.txt](https://github.com/abbystvns/gtv_forecasting/blob/master/requirements.txt). You can install them by
running `pip install requirements.txt` in your terminal. To avoid conflicts with existing Python environments, I suggest
doing this in a fresh virtual environment (see [here](https://docs.python-guide.org/dev/virtualenvs/) for details).

4. The worked example in this repository is in a [Jupyter Notebooks](https://jupyter.org/). After you've installed the appropriate 
dependencies, navigate to the repository and type `jupyter notebook`. A window should open in your browser that allows 
you to interact with the notebooks.

Within the Google Drive storing the data there are two folder: SSF and TRIPODS. These refer to two different climate-related
grants and provide different types of data.

#### Datasets
The dataset we used to predict winter precipitation over the Southwest United States (SWUS) using summer SSTs over the Pacific basin. 
The processed data in this folder consists of:

1. `X_obs.csv`: Monthly summer (July, Aug, Sept, Oct) SST observations from 1940-2019 at a 10 degree by 10 degree spatial scale.
More info/downloads [here](https://psl.noaa.gov/data/gridded/data.cobe2.html).

2. `sst_columns.csv`: Information about the columns of `X_obs.csv` (each row corresponds to a column in X).
in `X_obs.csv` (i.e the first row of `sst_loc.csv` gives you the location and month of the first column of `X_obs.csv`).

3. `y_avg.csv`: Winter (Nov - March) precipitation totals over the Southwestern US from 1941 to 2019. More info/downloads [here](https://www.ncdc.noaa.gov/cag/national/time-series).

4. `X_lens.csv`: Monthly summer (July, Aug, Sept, Oct) SST simulations from the CESM-LENS project at the same temporal and spatial granularity, 
but from 40 runs of the simulation (SST_lens.csv). Each simulation covers 66 years (1940 - 2005) so there are 2640 rows in this 
dataset where the first 66 rows correspond to the first simulation, the next 66 to the second simulations, and so on. 
These both include a "simulation" column to help keep track of things.
More info/downloads [here](http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html).
