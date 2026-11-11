# AI-Based College Placement Prediction System

## Project Overview
This project is an Artificial Intelligence and Machine Learning based College Placement Prediction System developed as a BCA Capstone Project.

The system uses student academic and other relevant information to predict placement outcomes using machine learning techniques.

## Objectives
- Predict student placement outcomes.
- Apply machine learning to educational data.
- Provide an easy-to-use web application.
- Help students understand factors related to placement opportunities.

## Technologies Used
- Python
- Machine Learning
- Flask (if used in the application)
- HTML, CSS
- Pandas
- NumPy
- Scikit-learn

## Dataset
The project uses a student placement dataset for model training and testing.

**Dataset location:** See the dataset folder in this repository.

## Project Structure
The repository contains the application source code, dataset, machine learning model files, templates, static resources and project documentation.

## How to Run
1. Install Python.
2. Install the required libraries using:

   `pip install -r requirements.txt`

3. Run the application using:

   `python app.py`

4. Open the local URL displayed in the terminal.

## Student Details
- **Student Name:** Mohit Kumar Thakur
- **Student ID:** 24BCAHH010008
- **Programme:** BCA
- **Institution:** ICFAI Tech

## Project Guide
Dr. Banoth Seetha Ramulu

## Disclaimer
This application is intended for academic and educational purposes. Predictions are estimates and should not be treated as guarantees of employment or placement.
# AI-Based College Placement Prediction System

BCA Capstone Project

## Overview
A Flask + scikit-learn web application that predicts a student's placement outcome from academic performance, skills, internships, certifications, projects, attendance, aptitude, communication, and interview performance.

## Quick Start
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open the local URL shown by Flask.

The project automatically creates a synthetic training dataset if `data/placement_data.csv` is missing and trains a Random Forest model.

## Important
The included dataset is synthetic/demo data. Replace it with an approved college dataset for an academic study, and remove personally identifiable information.

## Project Structure
- app.py - Flask application
- train_model.py - dataset generation and ML training
- data/placement_data.csv - demo dataset
- models/ - trained model and metrics
- templates/ - web pages
- static/style.css - styling
