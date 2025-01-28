import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns  
from imblearn.over_sampling import SMOTE  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.decomposition import PCA  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score  
from sklearn.metrics import recall_score  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import f1_score 
from sklearn.metrics import classification_report 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
# READING THE DATASET 
df = pd.read_csv(r'star_classification.csv') 
print(df) 
# HEAD, TAIL, INFO, DESCRIBE 
df.head() 
df.tail() 
df.info() 
df.describe() 
# LABEL ENCODING 
le = LabelEncoder()  
df['class'] = le.fit_transform(df['class'])  
df['class'].value_counts()  
# count plot of class  
sns.countplot(x='class', data=df) plt.show()  
# VARIABLE SEPARATION  
Y = df['class'] 
X = df.drop(columns=['class']) 
labels = set(Y) print(labels)  
# SCALERIZATION 
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
#SMOTENNING 
smote = SMOTE(random_state=42)  
X_res, Y_res = smote.fit_resample(X_scaled, Y) 
Y_res.value_counts() 
plt.hist(Y_res)  
# PCA (PRINCPLE COMPONENT ANALYSIS)  
pca = PCA(15)  
X_pca = pca.fit_transform(X_res) 
arr = np.array(X_pca) print(arr.shape)  
#TRAINING-TESTING-SPLITTING 
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y_res, test_size=0.2, random_state=42) 
print("X shape is", X.shape)  
print("Y shape is", Y.shape)  
print("X_test shape is", X_test.shape)  
print("X_train shape is", X_train.shape) 
print("y_train shape is", Y_train.shape) 
print("y_test shape is", Y_test.shape) 
# USING SVM CLASSIFIER  
svm = SVC()  
svm.fit(X_train, Y_train)  
Y_pred = svm.predict(X_test)  
print("Original y_test values are ", Y_test)  
print("predicted y_test values are", Y_pred)  
accuracy = accuracy_score(Y_test, Y_pred)  
precision = precision_score(Y_test, Y_pred, average='weighted') 
recall = recall_score(Y_test, Y_pred, average='weighted') 
f1 = f1_score(Y_test, Y_pred, average='weighted')  
print(f'Accuracy: {accuracy:.2f}')  
 print(f'Precision: {precision:.2f}')  
         print(f'Recall: {recall:.2f}')  
         print(f'F1 Score: {f1:.2f}')  
         conf_matrix = confusion_matrix(Y_test, Y_pred) 
         print(conf_matrix) 
         #HEATMAP OF CONFUSION MATRIX  
         plt.figure(figsize=(8, 6))  
         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues') 
         plt.xlabel('Predicted')  
         plt.ylabel('True') 
         plt.title('Confusion Matrix of SVM')  
         plt.show()  
         # CLASSIFICATION REPORT OF SVM 
         class_report=classification_report(Y_test,Y_pred,target_names=labels)                        
         print(class_report)  
         # USING K-NN CLASSIFIER  
         knn = KNeighborsClassifier()  
         knn.fit(X_train, Y_train) 
         Y_pred = knn.predict(X_test)  
         print("Original y_test values are ", Y_test) 
         print("predicted y_test values are", Y_pred) 
         accuracy = accuracy_score(Y_test, Y_pred)  
         precision = precision_score(Y_test, Y_pred, average='weighted')  
         recall = recall_score(Y_test, Y_pred, average='weighted')  
         f1 = f1_score(Y_test, Y_pred, average='weighted')  
         print(f'Accuracy: {accuracy:.2f}')  
         print(f'Precision: {precision:.2f}')  
         print(f'Recall: {recall:.2f}')  
         print(f'F1 Score: {f1:.2f}')  
         conf_matrix = confusion_matrix(Y_test, Y_pred)  
        print(conf_matrix)  
        # HEAT MAP OF CONFUSION MATRIX  
        plt.figure(figsize=(8, 6))  
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')  
        plt.xlabel('Predicted')   plt.ylabel('True')  
         plt.title('Confusion Matrix of KNN')  
         plt.show()  
         # CLASSIFICATION REPORT OF K-NN  
         class_report = classification_report(Y_test, Y_pred, target_names=labels)    
         print(class_report) 
         # USING NAIVE BAYES CLASSIFIER 
         model = GaussianNB() model.fit(X_train, Y_train) 
         Y_pred = model. predict(X_test) 
         print("Original y_test values are ", Y_test)  
         print("predicted y_test values are", Y_pred)  
         accuracy = accuracy_score(Y_test, Y_pred)  
         precision = precision_score(Y_test, Y_pred, average='weighted')  
         recall = recall_score(Y_test, Y_pred, average='weighted')  
         f1 = f1_score(Y_test, Y_pred, average='weighted')  
         print(f'Accuracy: {accuracy:.2f}')  
         print(f'Precision: {precision:.2f}')  
         print(f'Recall: {recall:.2f}')  
         print(f'F1 Score: {f1:.2f}')  
         conf_matrix = confusion_matrix(Y_test, Y_pred)  
         print(conf_matrix)  
         # CONFUSION MATRIX OF NAVIE BAYES  
         plt.figure(figsize=(8, 6))  
         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues') 
         plt.xlabel('Predicted') plt.ylabel('True')  
         plt.title('Confusion Matrix of Navie bayes')    plt.show()  
         # CLASSIFICATION REPORT OF NAVIE BAYES  
         class_report=classification_report(Y_test,Y_pred, target_names=labels) 
         print('Classification Report of Navie_bayes:')  
         print(class_report)  
        print(f'Precision: {precision:.2f}')  
         print(f'Recall: {recall:.2f}')  
         print(f'F1 Score: {f1:.2f}')  
         conf_matrix = confusion_matrix(Y_test, Y_pred)  
         class_report=classification_report(Y_test,Y_pred, target_names=labels) 
