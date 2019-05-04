# Convolution Neural Network Profiler  
A tool to profile the classification speed of a neural network across hardware platforms.  

This project attempts to make a 1:1 comparison of neural network classifiers across platforms.  It was initially made to compare Google Tensor Processing Unit classification, but can be used on any CPU or GPU installation of Tensorflow as well.  Profiling is basic, using Python's pstats package as a line magic in Jupyter.  Statistics and classification results are saved into `/data` and any files placed in that directory can be used to generate a comparison table.  

## Project Structure  

    |-- Compile Statistics.ipynb                    # notebook for comparing profiles
    |-- config.py                                   # project configuration variables
    |-- confusion.py                                # confusion matrix formatter
    |-- data                                        # output data profiles and image classfications
    |-- download.py                                 # methods for downloaded required data
    |-- downloads                                   # folder downloaded data is saved to
    |-- environment_cpu.yml                         # Conda CPU environment file
    |-- environment_gpu.yml                         # Conda GPU environment file    
    |-- flow.py                                     # Keras ImageDataGenerator methods
    |-- ImageNet Dog Data Download.ipynb            # Notebook to download required data
    |-- LICENSE                                     # Project license
    |-- Dog Classifcation Profiler.ipynb            # Notebook to run classifications and profile from
    |-- models.py                                   # Tensorflow model building methods
    |-- pstats_parser.py                            # Parser for pstats output
    |-- README.md                                   # This file
