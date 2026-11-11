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
