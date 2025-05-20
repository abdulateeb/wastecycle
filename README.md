## wastecycle

## WASTE CLASSIFICATION SYSTEM FOR AUTOMATED SORTING IN MATERIAL RECOVERY FACILITIES


## Project Overview
This project aims to build an intelligent machine learning model to classify waste into seven categories — Cardboard, E-Waste, Glass, Medical, Metal, Paper, and Plastic — by analyzing images of waste materials.
The goal is to implement this system into Material Recovery Facilities (MRFs) to replace manual waste sorting with an automated real-time classification system working over a conveyor belt.

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model on your dataset:
```bash
python train.py
```

### Running the Web Application

To start the Streamlit web interface:
```bash
streamlit run app.py
```

The web interface allows you to upload images and get real-time predictions for waste classification.

---

## Modules

- **Data Processing and Preprocessing**  
  Image resizing, normalization, and augmentation to improve model performance.

- **Model Training**  
  Training a machine learning model to classify images into the 7 categories.

- **Web Application Deployment**  
  Streamlit-based webapp for testing model where users can upload and classify waste images.

- **Evaluation and Testing**  
  Real-world testing on unseen data with accuracy.

---

## Project Objective

To automate waste classification at Material Recovery Facilities (MRFs) by replacing manual sorting using an intelligent ML model capable of real-time classification of mixed waste on conveyor belts.

---

## Team Members

| Name                   | Hall Ticket Number  |
|------------------------|---------------------|
| Abdul Ateeb            | 217021026144        |
| Syeeda Zoya Tabassum   | 217021026047        |

---

## Acknowledgements

This project is done as part of the final year major project for the BS-MS (Computer Science) program at Osmania University.

---