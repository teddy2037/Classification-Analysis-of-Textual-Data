## Project 1

Each file in this folder contains it's own main function that can be used to test the functionality of the script. Each file roughly contains the solution to one question: apart from Q1,2 which are both in the same file. Also note that the files must be under one folder. The scripts reference one another.

*We recommend that you open this README in another tab as you explore this folder. This project is authored by Pavan S Holur, Ravi Teja, Megan Williams and Donna Branchevsky. It was created for ECE 219 - Large-Scale Data Mining 2019 @ UCLA.*

---

**Running the Code**

To start, please download the entire repository with all python files and readme. The results can be found in the report created from the data used in the project.
How to run each file:

1. Please run Question 1 and 2 BEFORE running the other files. These functions install NLTK packages if not already present, and future solutions assume the environment to be set up.
2. The remaining files can be run in any order. Note that, starting from q4, the functions CALL q2,3 inherently.

To run a script in python, simply run the following command: 
(verify that all libraries included in our files have been installed: with pip, do $ pip install <>)

$ python Project1_q*.py
	where (*) denotes the various files in the folder.
	
To pipeline, Project1_q7.py contains code that generates the Excel spreadsheet using Pandas. The spreadsheet can be used to compare various classifiction chains.
Note: The headers / footers vs. none option is changed manually in the script to generate two spreadsheets which are compared.
