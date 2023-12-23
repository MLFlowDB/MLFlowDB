# MLFlowDB Lite Implementation

This directory includes a streamlined implementation of MLDR (MLFlowDB Lite), which is capable of realizing the basic functionalities of MLFlowDB as described in our paper.

## Getting the code

You can download a copy of all the files in this repository by cloning the [git]([MLFlowDB/MLFlowDB (github.com)](https://github.com/MLFlowDB/MLFlowDB)) repository:

```
git clone https://github.com/MLFlowDB/MLFlowDB.git
```

## Dependencies

You'll need a working Python environment to run the code. The recommended way to set up your environment is through the [Anaconda Python distribution](https://gitee.com/link?target=https%3A%2F%2Fwww.anaconda.com%2Fdownload%2F) which provides the `conda` package manager. Anaconda can be installed in your user directory and does not interfere with the system Python installation. The required dependencies are specified in the file `mlflowdb_environment.yaml`.

We use `conda` virtual environments to manage the project dependencies in isolation. Thus, you can install our dependencies without causing conflicts with your setup (even with different Python versions).

Run the following command in the repository folder (where `mlflowdb_environment.yaml.yml` is located) to create a separate environment and install all required dependencies in it:

```
conda env create
```

## Re-running

Before running any code you must activate the conda environment:

```
conda activate mldr
```

This will enable the environment for your current terminal session. Any subsequent commands will use software that is installed in the environment.

**Ensure that MongoDB are installed, and configure the addresses, ports, usernames, passwords, and databases for MongoDB  in `MLDRSys/settings.py`.**.

To run MLFlowDB, run this in the top level of the repository:

```
python3 manage.py runserver 127.0.0.1:8000
```

## MLFlowDB VS. MLFlowDB Lite 

The actual implementation of MLFlowDB is a more complex version. It builds upon MLDR with additional considerations for performance and usability. The official version of MLFlowDB is more akin to a complete machine learning platform rather than just a lifecycle management tool. Compared to MLFlow Lite, the MLFlowDB official version also possesses the following characteristics:

*   Implements a tiered storage approach for Origin Data, compressing old JSON data into columnar storage on HDFS, and utilizes Parquet Arrow for read and write operations.
*   Supports online execution of Pandas data transformation scripts, recording data transfer processes in fine detail within the platform environment, and facilitates reuse.
*   Provides a range of preprocessing methods for semi-structured JSON data, including JSON concatenation, JSON dimensionality reduction, etc.

We plan to open source the complete and mature version of MLFlowDB later. Until then, you can still use MLFlowDB Lite for querying, downloading, uploading, and tracking elements and relationships, as it already encompasses the objectives outlined in our paper.