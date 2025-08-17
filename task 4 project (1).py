#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy scikit-learn matplotlib seaborn')




# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

heart_df = pd.read_csv("heart (1).csv")
X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
scaler_heart = StandardScaler()
Xh_train_scaled = scaler_heart.fit_transform(Xh_train)
Xh_test_scaled = scaler_heart.transform(Xh_test)

heart_model = RandomForestClassifier()
heart_model.fit(Xh_train_scaled, yh_train)


# In[3]:


cancer_df = pd.read_csv("breast-cancer (1).csv")

# Drop ID column and map target
cancer_df.drop(columns='id', inplace=True)
cancer_df['diagnosis'] = cancer_df['diagnosis'].map({'M': 1, 'B': 0})

X_cancer = cancer_df.drop("diagnosis", axis=1)
y_cancer = cancer_df["diagnosis"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)
scaler_cancer = StandardScaler()
Xc_train_scaled = scaler_cancer.fit_transform(Xc_train)
Xc_test_scaled = scaler_cancer.transform(Xc_test)

cancer_model = RandomForestClassifier()
cancer_model.fit(Xc_train_scaled, yc_train)


# In[4]:


diabetes_df = pd.read_csv("diabetes.csv")
X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
scaler_diabetes = StandardScaler()
Xd_train_scaled = scaler_diabetes.fit_transform(Xd_train)
Xd_test_scaled = scaler_diabetes.transform(Xd_test)

diabetes_model = RandomForestClassifier()
diabetes_model.fit(Xd_train_scaled, yd_train)


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 4))

# Heart
plt.subplot(1, 3, 1)
sns.countplot(data=heart_df, x='target', palette='Set2')
plt.title("Heart Disease Classes")

# Cancer
plt.subplot(1, 3, 2)
sns.countplot(data=cancer_df, x='diagnosis', palette='Set1')
plt.title("Breast Cancer (M=1 / B=0)")

# Diabetes
plt.subplot(1, 3, 3)
sns.countplot(data=diabetes_df, x='Outcome', palette='Set3')
plt.title("Diabetes Outcome Classes")

plt.tight_layout()
plt.show()


# In[6]:


plt.figure(figsize=(12, 8))
sns.heatmap(heart_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("🔴 Heart Disease - Feature Correlation Heatmap")
plt.show()


# In[7]:


plt.figure(figsize=(14, 10))
sns.heatmap(cancer_df.corr(), annot=True, fmt=".2f", cmap="Purples", square=True)
plt.title("🟣 Breast Cancer - Feature Correlation Heatmap")
plt.show()


# In[8]:


plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_df.corr(), annot=True, fmt=".2f", cmap="YlGnBu", square=True)
plt.title("🟢 Diabetes - Feature Correlation Heatmap")
plt.show()


# In[9]:


import tkinter as tk
from tkinter import ttk, messagebox

# GUI Root Window
root = tk.Tk()
root.title("Disease Prediction System")
root.geometry("500x600")
root.configure(bg="#f5f5f5")

# ----- STYLE SECTION -----
style = ttk.Style()
style.theme_use("clam")

# Notebook tab style
style.configure("TNotebook", background="#f5f5f5", borderwidth=0)
style.configure("TNotebook.Tab", font=('Helvetica', 12, 'bold'), padding=[10, 5])
style.map("TNotebook.Tab", background=[("selected", "#e0e0e0")])

# Label and Entry style (ttk versions)
style.configure("TLabel", font=("Helvetica", 10), background="#f5f5f5")
style.configure("TEntry", padding=5)

# Button style
style.configure("TButton", font=("Helvetica", 10, "bold"), padding=6)
# -------------------------

notebook = ttk.Notebook(root)

# Create 3 Tabs
tabs = [tk.Frame(notebook, bg="#f5f5f5") for _ in range(3)]
notebook.add(tabs[0], text='Heart Disease')
notebook.add(tabs[1], text='Breast Cancer')
notebook.add(tabs[2], text='Diabetes')
notebook.pack(expand=True, fill='both', padx=10, pady=10)

# Generic form builder
def build_form(tab, features, model, scaler, title):
    entries = []
    for i, feat in enumerate(features):
        ttk.Label(tab, text=feat).grid(row=i, column=0, padx=10, pady=3, sticky='w')
        e = ttk.Entry(tab, width=25)
        e.grid(row=i, column=1, padx=10, pady=3)
        entries.append(e)

    def predict():
        try:
            data = [float(e.get()) for e in entries]
            scaled = scaler.transform([data])
            result = model.predict(scaled)[0]
            label = "🟢 No Disease" if result == 0 else "🔴 Disease Detected"
            messagebox.showinfo(f"{title} Prediction", label)
        except Exception as ex:
            messagebox.showerror("Input Error", f"Check your inputs!\n\n{ex}")

    ttk.Button(tab, text="Predict", command=predict).grid(row=len(features), columnspan=2, pady=15)

# Build all 3 tabs
build_form(tabs[0], X_heart.columns, heart_model, scaler_heart, "Heart Disease")
build_form(tabs[1], X_cancer.columns, cancer_model, scaler_cancer, "Breast Cancer")
build_form(tabs[2], X_diabetes.columns, diabetes_model, scaler_diabetes, "Diabetes")

root.mainloop()


# In[ ]:




