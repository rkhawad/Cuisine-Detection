import sys
import json
from nltk import wordnet as wn
import editdistance

def parse_json_and_dump(json_file, output_file, feature_list, cuisine_list):
  with open(json_file) as data_file:    
    data = json.load(data_file)


  string_split = lambda x: x.split(' ')
  Synsets = lambda x: wn.wordnet.synsets(x, pos=wn.wordnet.NOUN)
  filter_lists = lambda x: len(x) != 0
  synset_filter = lambda x: x[0]
  lemmas = lambda x: [str(lemma.name()) for lemma in x.lemmas()]

  feature_ingredients = []
  cuisine_outputs = []
  index = 0
  for data_set in data:
    indey = 0
    ingredients_list = data_set["ingredients"]
    if cuisine_list != "":
      cuisine_outputs.append(data_set["cuisine"])
    for ingredient in ingredients_list:
      ingredient = ingredient.replace(' ', '_')
      Synsets_filter = map(synset_filter, filter(filter_lists, map(Synsets, [ingredient])))
      lemma_name = " ".join(map(synset_filter, map(lemmas, Synsets_filter)))
      if lemma_name == "":
        ingredients = ingredient.split('_')
        Synsets_filter = map(synset_filter, filter(filter_lists, map(Synsets, ingredients)))
        lemma_name = " ".join(map(synset_filter, map(lemmas, Synsets_filter))) 
      print lemma_name
      lemma_name = lemma_name.lower()
      if feature_list != "":
        feature_ingredients.append(lemma_name)
      data[index]["ingredients"][indey] = lemma_name
      indey += 1
    index += 1

  with open(output_file, 'w') as outfile:
    json.dump(data, outfile)
  if feature_list == "":
    return
  feature_set = set(feature_ingredients)
  cuisine_set = set(cuisine_outputs)

  f = open(feature_list, 'w')
  index = 0
  for item in feature_set:
    f.write(item + "\n")
  f.close()

  f = open(cuisine_list, 'w')
  for item in cuisine_set:
    f.write(item + "\n")
  f.close()

if __name__ == "__main__":
  parse_json_and_dump(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
  parse_json_and_dump(sys.argv[5], sys.argv[6], "", "")
