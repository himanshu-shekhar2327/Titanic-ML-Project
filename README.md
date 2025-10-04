# 🛳️ Titanic - Machine Learning from Disaster  

##  Overview  
This project is based on the famous **Kaggle competition** — [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).  
The goal of this project is to **predict which passengers survived the Titanic shipwreck** using data analysis and machine learning techniques.  

The project is divided into two main parts:  
1. **Exploratory Data Analysis (EDA)**  
2. **Model Building & Evaluation**

---

##  Problem Statement  
The sinking of the Titanic is one of the most infamous shipwrecks in history.  
In this challenge, the objective is to **build a predictive model** that answers the question:  
> “What sorts of people were more likely to survive?”  

Using passenger data (such as name, age, gender, socio-economic class, etc.), the model aims to predict survival outcomes.

---

##  Dataset  
The dataset used in this project was obtained from **Kaggle** and contains the following features:  

| Feature | Description |
|----------|--------------|
| PassengerId | Unique ID for each passenger |
| Survived | Survival (0 = No, 1 = Yes) |
| Pclass | Ticket class (1 = Upper, 2 = Middle, 3 = Lower) |
| Name | Passenger’s name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

##  Part 1: Exploratory Data Analysis (EDA)
In the **EDA Notebook**, I explored the dataset to uncover patterns and relationships.  
**Steps performed:**
- Data inspection and missing value analysis  
- Visualization of survival rate by gender, class, and age  
- Feature correlations and insights  
- Outlier detection and handling  
- Feature engineering (creating new features like `FamilySize`, `IsAlone`, etc.)

---

##  Part 2: Model Building & Evaluation  
In the **Model Building Notebook**, multiple models were trained and evaluated to predict survival:  
**Steps performed:**
- Encoding categorical variables  
- Feature scaling  
- Splitting data into training and testing sets  
- Training models by **Logistic Regression**  
- Hyperparameter tuning using GridSearchCV  
- Model evaluation using accuracy.  
- Saving the best-performing model using `joblib`  

---

##  Key Insights  
- **Females** had a much higher survival rate than males.  
- **1st class passengers** had better chances of survival than lower classes.  
- **Passengers traveling alone** had a lower probability of survival.  
- **Age and Fare** played significant roles in survival prediction.  

---
## Model Performance 
- After splitting the training dataset for model evaluation:
- Training Accuracy: 83.14%
- Validation Accuracy: 81.56%
Finally, predictions were made on the Kaggle test dataset, and the submission file was generated in the required format for the competition.

## Kaggle Submission
The final model was used to predict survival outcomes for passengers in the test dataset.
The submission file was uploaded to Kaggle under the competition “Titanic: Machine Learning from Disaster.”

##  Technologies Used  
- **Python**  
- **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn**  
- **Scikit-learn**  
- **Joblib** (for model serialization)  
- **Jupyter Notebook**

---

## 📂 Project Structure  
Titanic-ML-Project/
│
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│
├── notebooks/
│   ├── Titanic_EDA.ipynb      # Exploratory Data Analysis (EDA)
│   ├── Model_Building.ipynb   # Model training, evaluation, and Kaggle submission
│
├── titanic_model.pkl          # Saved trained model (Logistic Regression pipeline)
├── submission.csv             # Final prediction file for Kaggle submission
├── README.md                  # Project documentation
├── requirements.txt           # Required Python packages


---
## How to Run the Project
**1.Clone the repository**
git clone https://github.com/himanshu-shekhar2327/Titanic-ML-Project.git
cd Titanic-ML-Project

**2.Install dependencies**
pip install -r requirements.txt

**3.Open Jupyter Notebook**
jupyter notebook


##  Results
- Model Used: Logistic Regression
- Train Accuracy: 83.14%
- Test Accuracy: 81.56%
- Kaggle Public Score: 0.76315


##  Future Enhancements  
- Build a **Streamlit web app** for user interaction  
- Deploy the model using **FastAPI** or **Flask**  
- Perform advanced feature engineering  
- Add cross-validation and ROC curve analysis  

---

##  Acknowledgements  
- [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
- Kaggle Community for insights and inspiration  


