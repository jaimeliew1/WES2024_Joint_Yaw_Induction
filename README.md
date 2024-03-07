
# WES2024

This repository contains the code used to generate data and create figures for the upcoming research paper titled ***Joint Yaw-Induction Control for Wind Farms***, which is to be submitted to the journal of Wind Energy Science. The structure and usage of the repository is outlined below.


## Getting Started

1. Clone this repository:

```bash
git clone https://github.com/jaimeliew1/WES2024_Joint_Yaw_Induction.git
cd WES2024_Joint_Yaw_Induction
```

2. Install [Poetry](https://python-poetry.org/) (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install the necessary dependencies:

```bash
poetry install
```

4. Run the data generation module:

```bash
./scripts/generate_all.sh
```

5. Run the plotting scripts:

```bash
./scripts/plot_all.sh
```


## Modules

### 1. Generate (Module: WES2024/Generate)

This module contains the code responsible for generating the data required for the research paper. The data is generated the first time these functions are run and is subsequently cached for future use. The generated data is stored in a folder named `data` as `.csv` files. Please note that running the data generation module may require a high CPU count system and can take several hours.



### 2. Plotting (Module: WES2024/Plot)

This module includes scripts for creating figures used in the research paper. Each script corresponds to a specific figure in the paper. The plotting scripts call relevant functions from the Generate module. If the required data has been generated before, the plotting script retrieves it; otherwise, it triggers the data generation process.



## Folder Structure

- `WES2024/Generate`: Contains code for data generation.
  - ...

- `WES2024/Plot`: Contains scripts for creating figures.
  - ...

- `data`: Folder to store generated data.

