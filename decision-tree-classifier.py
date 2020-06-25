#python decision-tree-classifier.py -t ./labeled-data-samples/access.csv -v ./labeled-data-samples/access.2020-04-28.csv

from utilities import *

args = get_args()
traning_data = args['traning_data']
testing_data = args['testing_data']

# Get training features and labeles
training_features, traning_labels = get_data_details(traning_data)

# Get testing features and labels
testing_features, testing_labels = get_data_details(testing_data)

# DECISON TREE CLASSIFIER
print("\n\n=-=-=-=-=-=-=- Decision Tree Classifier -=-=-=-=-=-=-=-\n")

# Instanciate the classifier
attack_classifier = tree.DecisionTreeClassifier()

# Train the classifier
attack_classifier = attack_classifier.fit(training_features, traning_labels)

# Get predections for the testing data
predictions = attack_classifier.predict(testing_features)

print("The precision of the Decision Tree Classifier is: " + str(get_occuracy(testing_labels,predictions, 1)) + "%")
