import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn import linear_model,preprocessing

data = pd.read_csv("car.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
persons = le.fit_transform(list(data["persons"]))
door = le.fit_transform(list(data["door"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))
safety = le.fit_transform(list(data["safety"]))

predict = "class"
x = list(zip(buying, maint, persons, lug_boot, safety, door))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("predicted:", names[predicted[x]], "data:", x_test[x], "actual:", names[y_test[x]])