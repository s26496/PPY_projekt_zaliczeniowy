import pandas as pd
import joblib
import numpy as np
from tkinter import *
from tkinter import messagebox, scrolledtext, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import sqlite3



class CarEvaluationApp:

    def __init__(self, master):
        self.master = master
        master.title("Car Evaluation App")

        # ladowanie i kodowanie danych
        self.load_data()

        # model
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # UI
        self.label = Label(master, text="Enter new data:")
        self.label.pack()

        self.entry = Entry(master, width=100)
        self.entry.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict_new_data)
        self.predict_button.pack()

        self.new_data_button = Button(master, text="Add new data", command=self.add_new_data)
        self.new_data_button.pack()

        self.retrain_button = Button(master, text="Rebuild model", command=self.retrain_model)
        self.retrain_button.pack()

        self.save_model_button = Button(master, text="Save Model", command=self.save_model)
        self.save_model_button.pack()

        # button do wizualizacji na grafie
        self.visualize_button = Button(master, text="Visualize Data", command=self.visualize_data)
        self.visualize_button.pack()

        # buttony do zapisu/wczytania danych z bazy
        self.save_to_db_button = Button(master, text="Save Data to DB", command=self.save_data_to_db)
        self.save_to_db_button.pack()

        self.load_from_db_button = Button(master, text="Load Data from DB", command=self.load_data_from_db)
        self.load_from_db_button.pack()

        # tabela do wyswietlania danych
        self.tree = ttk.Treeview(master, columns=list(self.data.columns), show="headings")
        for col in self.data.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack()

        # wprowadzenie danych do tabeli
        for index, row in self.data.iterrows():
            self.tree.insert("", "end", values=list(row))

        self.report_area = scrolledtext.ScrolledText(master, wrap=WORD, width=100, height=10)
        self.report_area.pack()

        # ladowanie/trenowanie modelu
        if not self.load_model():
            self.train_model()

    # ladowanie danych
    def load_data(self):
        data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        attribute_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        self.data = pd.read_csv(data_url, header=None, names=attribute_names)
        self.encoder = OneHotEncoder(drop='first')
        self.X = self.encoder.fit_transform(self.data.drop('class', axis=1)).toarray()
        self.y = self.data['class']

    # trenowanie danych
    def train_model(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.clf.fit(X_train, self.y_train)
        y_pred = self.clf.predict(X_test)

        # wstawianie dokładności modelu i raportu klasyfikacji do text area
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        self.report_area.insert(INSERT, f"Model Accuracy: {accuracy}\n\nClassification Report:\n{report}\n")

    # predykcja nowo wprowadzonych danych
    def predict_new_data(self):
        new_data_str = self.entry.get()
        try:
            new_data = [new_data_str.split(",")]
            new_data_encoded = self.encoder.transform(new_data).toarray()
            prediction = self.clf.predict(new_data_encoded)
            messagebox.showinfo("Prediction", f"The predicted class is: {prediction[0]}")
        except:
            messagebox.showerror("Error", "Invalid input format or model not trained")

    # retrenowanie modelu
    def retrain_model(self):
        try:
            self.report_area.delete(1.0, END)  # czyszczenie text area
            self.train_model()
            messagebox.showinfo("Success", "Model rebuilt successfully")
            self.save_model()
        except:
            messagebox.showerror("Error", "Failed to rebuild the model")

    # zapis modelu do pliku
    def save_model(self):
        try:
            joblib.dump(self.clf, 'model.pkl')
            messagebox.showinfo("Success", "Model saved successfully")
        except:
            messagebox.showerror("Error", "Failed to save the model")

    # wczytanie modelu z pliku
    def load_model(self):
        try:
            self.clf = joblib.load('model.pkl')
            return True
        except:
            return False

    #
    def add_new_data(self):
        new_data_str = self.entry.get()
        try:
            new_data = new_data_str.split(",")
            new_row = pd.DataFrame([new_data], columns=self.data.columns[:-1])
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.X = self.encoder.fit_transform(self.data.drop('class', axis=1)).toarray()
            messagebox.showinfo("Success", "New data added successfully")

            # wprowadzenie nowych danych do treeview
            self.tree.insert("", "end", values=new_data)

        except:
            messagebox.showerror("Error", "Invalid input format")

    def visualize_data(self):
        # policz wystapienia kazdej klasy
        class_counts = self.data['class'].value_counts()

        # graf
        plt.figure(figsize=(8, 4))
        plt.bar(class_counts.index, class_counts.values)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Count of Each Class in the Car Evaluation Dataset')
        plt.show()

    def save_data_to_db(self):
        try:
            # Connect to SQLite database
            conn = sqlite3.connect("car_evaluation.db")
            # Save data to SQLite database
            self.data.to_sql("car_data", conn, if_exists="replace", index=False)
            # Close connection
            conn.close()
            messagebox.showinfo("Success", "Data saved to database successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data to database\n{e}")

    def load_data_from_db(self):
        try:
            # Connect to SQLite database
            conn = sqlite3.connect("car_evaluation.db")
            # Load data from SQLite database
            self.data = pd.read_sql("SELECT * FROM car_data", conn)
            # Close connection
            conn.close()
            messagebox.showinfo("Success", "Data loaded from database successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data from database\n{e}")


root = Tk()
app = CarEvaluationApp(root)
root.mainloop()
