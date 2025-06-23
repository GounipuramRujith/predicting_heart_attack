

       💓 Heart Attack Prediction System

This project predicts the risk of a heart attack using two models:

1. **CNN (Convolutional Neural Network)** – For ECG image-based prediction.
2. **Random Forest Classifier** – For symptom and ECG parameter-based prediction using tabular data.

It provides both a CLI interface (no Streamlit) and CSV-based workflow for data scientists or medical teams.

---

## 📁 Project Structure

```

heart\_attack\_predictor/
│
├── app.py                      # (optional) Main Streamlit app (if using UI)
├── cnn\_model.py                # Code for loading/training CNN model
├── symptom\_model.py            # Code for symptom-based prediction
├── train\_rf\_model.py           # Script to train RF model on full dataset
├── predict\_rf\_csv.py           # Predicts using user\_input.csv (no manual input)
├── requirements.txt            # Required Python packages
├── model/
│   ├── cnn\_ecg\_model.h5        # Pretrained CNN model
│   └── symptom\_model.pkl       # Trained Random Forest model
├── merged\_heart\_attack\_dataset.csv  # Full dataset for training
├── user\_input.csv              # One-row test data for prediction

````

---

## 🚀 How It Works

### 1. CNN ECG Image Model
- Trained to classify ECG images into high-risk or low-risk.
- Input: ECG image (`.jpg`, `.png`)
- Output: Heart attack risk prediction

### 2. Random Forest Model
- Trained on tabular data (`age`, `heart_rate`, `qrs`, `qt`, `symptoms`, etc.)
- Input: One-row CSV of user values
- Output: Heart attack risk classification

---

## 🛠️ How To Run (Without UI)

### Step 1: Train Model
```bash
python train_rf_model.py
````

### Step 2: Predict with CSV

Make sure `user_input.csv` contains **exactly one row** with 13 feature columns.

Then run:

```bash
python predict_rf_csv.py
```

### Output:

```
✅ No signs of heart attack. (Confidence: 91.00%)
```

---

## 📄 Sample: user\_input.csv

```csv
age,heart_rate,qrs_duration,qtc,pr,chest_pain,shortness_of_breath,sweating,nausea,fatigue,dizziness,jaw_pain,shoulder_pain
55,90,110,430,145,1,1,0,0,1,0,0,1
```

---

## 💡 Future Enhancements

* Add Docker support for deployment
* Integrate Streamlit for interactive web UI
* Add REST API support with Flask or FastAPI
* Use real-time ECG signal capture

---

## 📦 Requirements

Install all packages with:

```bash
pip install -r requirements.txt
```

Packages include:

* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow`
* `Pillow`

