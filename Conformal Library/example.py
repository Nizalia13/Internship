import pandas as pd

import matplotlib.pyplot as plt

from conformal_envelopes import ConformalSetModel

from conformal_envelopes import evaluate_prediction_sets

train = pd.read_csv("train_example.csv")
test = pd.read_csv("test_example.csv")

print(train.head())
print()
print(test.head())

# ###### COLLAPSED MODEL ######
# # creating the model
# model = ConformalSetModel(method="collapsed", alpha=0.1, random_state=13)

# # Fitting the model using training file
# # takes the training data, separates the observations by class (A and B), splits each class into S1 and S2, 
# # learns the Collapsed envelope shape from S1, and calibrates it using S2.
# model.fit(train)

# # inspecting the constructed envelopes
# print(model.get_envelopes())

# # using the learnt envelopes to make the conformal prediction sets for the test data
# predictions = model.predict(test)
# print(predictions)

# # comparing each set with the true label in the test data
# results = evaluate_prediction_sets(predictions["prediction_set"], test["label"])
# print(results)


# ###### RADIAL MODEL ######
# # creating the model
# model = ConformalSetModel(method="radial", alpha=0.1, random_state=13,)

# # Fitting the model using training file
# # takes the training data, separates the observations by class (A and B), splits each class into S1 and S2, 
# # learns the Collapsed envelope shape from S1, and calibrates it using S2.
# model.fit(train)

# # inspecting the constructed envelopes
# print(model.get_envelopes())

# # using the learnt envelopes to make the conformal prediction sets for the test data
# predictions = model.predict(test)
# print(predictions)

# # comparing each set with the true label in the test data
# results = evaluate_prediction_sets(predictions["prediction_set"], test["label"])
# print(results)



###### STRIP MODEL ######
# creating the model
model = ConformalSetModel(method="strip", alpha=0.1, random_state=13,)

# Fitting the model using training file
# takes the training data, separates the observations by class (A and B), splits each class into S1 and S2, 
# learns the Collapsed envelope shape from S1, and calibrates it using S2.
model.fit(train)

# inspecting the constructed envelopes
print(model.get_envelopes())

# using the learnt envelopes to make the conformal prediction sets for the test data
predictions = model.predict(test)
print(predictions)

# comparing each set with the true label in the test data
results = evaluate_prediction_sets(predictions["prediction_set"], test["label"])
print(results)

# plotting 2D
model.plot(x="score_2", y="score_3", label="B")
plt.show()

