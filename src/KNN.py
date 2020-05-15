from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assigning features and label variables
# First Feature
weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
# Label or target variable
play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

print('Weather',len(weather))
print('Temperature',len(temp))
print('Label',len(play))

le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)
print('Weather Encoded',weather_encoded)

# Converting string labels into numbers.
temp_encoded = le.fit_transform(temp)
print('Temperature Encoded',temp_encoded)

label=le.fit_transform(play)
print('Label Encoded',label)

features = list(zip(weather_encoded,temp_encoded))
print('Features',features)

# Splitting train : test to 80 : 20 ratio
X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print('Predicted',y_pred)
print('Actual data',y_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy',accuracy)

