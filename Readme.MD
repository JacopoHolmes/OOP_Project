# Final project for the Object-Oriented Programming course

This repository contains the final project for the OOP course at the University of Ferrara.
The main goal of the course was to learn the basics of the OOP paradigm and how to apply it in C++ and Python.

The project consists of two different assignments, here briefly summarized (the entire assignment text is in the \Notes folder):
- **Claro**: Read all the files in the folder "secondolotto_1", find the good ones, extract the data about the transition point, width and store the ADC and Counts into two arrays. Via the Least Square Method on the transition zone (counts != 0 && counts !=1000) evaluate the transition point and compare it with the one present in the file. Use the erf function (SciPy module) to fit the points and evaluate again the transition point and width. Graph an histogram of the discrepancies between the expected and evaluated values.
- **SiPM**: Evaluate the direct and reverse curves of the SiPM data in the "CACTUS_HPK" folder. Analyze the peaks of each waveform. Estimate the dark count rate.


## Repository organization
- **\Claro**: contains the code for the claro assignment, with "claro_main.py" being the final program to execute and "claro_class.py" containing all the classes definitions. Also contains the output file folder and a (smaller) folder of input files
- 
- **\SiPM**: Same structure as \Claro but with the input, output and code for the SiPM assignment 
