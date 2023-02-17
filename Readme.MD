# Final project for the Object-Oriented Programming course

This repository contains the final project for the OOP course at the University of Ferrara.
The main goal of the course was to learn the basics of the OOP paradigm and how to apply it in C++ and Python.

The project consists of two different assignments:
- **Claro**: Read all the files in the folder "secondolotto_1", find the good ones, extract the data about the transition point, width and store the ADC and Counts into two arrays. Via the Least Square Method on the transition zone (counts != 0 && counts !=1000) evaluate the transition point and compare it with the one present in the file. Use the erf function (SciPy module) to fit the points and evaluate again the transition point and width. Graph these fits for a single file along with a histogram of the discrepancies between the expected and evaluated values of all the data in the folder.
- **SiPM**: For each subfolder in the "CACTUS_HPK" directory, evaluate the forward and reverse IV curves of the SiPM data. For the forward curves evaluate the Quenching resistance by fitting a linear model on the last part of the curve itself; for the reverse ones, evaluate the Breakdown voltage by fitting a 5th degree polynomial on the derivative of the data and evaluating its peak position. Plot histograms of these values and compare the two liquid nitrogen measures, as well as the one took on the same day.


## Repository organization
- **\Claro**: contains the code for the claro assignment, with "claro_main.py" being the final program to execute and "claro_class.py" containing all the classes definitions. Also contains the output file folder and a (smaller) folder of input files;
- **\SiPM**: same structure as \Claro but with the input, output and code for the SiPM assignment;
- **OOP_Report.pdf**: IN ITALIAN, brief summary of the code usage and outputs.


## Usage
Download the *_main.py and *_class.py and the requirements.txt files of each project and store them in a folder.
First run the following command to ensure that all the required packages are installed and up to date:
```pip install -r .\requirements.txt```

The *_main.py file is the "front end" of the program, change the various function arguments here and then run the program via terminal, specifying the file/directory path as the first argument.
