import sys
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer

f = open(sys.argv[1], 'r')
cuisine_list = map(lambda x : x.strip(), f.readlines())
f.close()

f = open(sys.argv[2], 'r')
ingredients_list = map(lambda x: x.strip(), f.readlines())
f.close()

cuisine_indices = [i for i in range(len(cuisine_list))]
ingredients_indices = [i for i in range(len(ingredients_list))]


cuisine_map = dict(zip(cuisine_list, cuisine_indices))
ingredients_map = dict(zip(ingredients_list, ingredients_indices))

cuisine_indices = []
ingredients_indices = []

with open(sys.argv[3]) as data_file:    
  data = json.load(data_file)

Y = []
X = []
for data_set in data:
  ingredients = []
  Y.append(cuisine_map[data_set["cuisine"]])
  for ingredient in data_set["ingredients"]:
    ingredients.append(ingredients_map[ingredient])
  X.append(ingredients)

X = np.array(X)
Y = np.array(Y)

#Training the model
model = BernoulliNB()


binarizer = MultiLabelBinarizer()
binarizer.fit([range(len(ingredients_list))])
x = binarizer.transform(X[:4000])
model.partial_fit(x, Y[:4000], classes=np.unique(Y))
X = X[4000:]
Y = Y[4000:]
while len(X) > 4000:
  x = binarizer.transform(X[:4000])
  model.partial_fit(x, Y[:4000])
  X = X[4000:]
  Y = Y[4000:]

x = binarizer.transform(X)
model.partial_fit(x, Y)

X = []
Y = []

with open(sys.argv[4]) as data_file:    
  output = json.load(data_file)
for data_set in output:
  ingredients = []
  for ingredient in data_set["ingredients"]:
    if ingredient in ingredients_map:
      ingredients.append(ingredients_map[ingredient])
  X.append(ingredients)

predicted = model.predict(binarizer.transform(X))
f = open(sys.argv[5], 'w')
matches = 0
for prediction in predicted:
  f.write(cuisine_list[prediction] + "\n")
  if cuisine_list[prediction] == "italian":
    matches += 1
f.close()


print "The number of matched cuisine ouputs are %d out of %d test sets" %(matches,  len(predicted))
