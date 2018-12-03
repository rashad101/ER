import json
from thesaurus import Word
from collections import Counter
import textdistance
import numpy as np

def count_ontology_classes():
    with open('../data/FullyAnnotated_LCQuAD_new.json') as f:
        data = json.load(f)
        no_defined_class = []
        classes = []
        print(len(data))
        total_q = 0

        for i in range (len(data)):
            for dict in data[i]:
                total_q+= len(data[i])
                question_tokens = dict["token question stop words"]
                #print("Question token = {}".format(question_tokens))
                tokens = question_tokens.split(" ")
                ontology_class = [c for c in tokens if c.isupper()]

                if len(ontology_class) == 0:
                    no_defined_class.append(question_tokens)
                else:
                    #print("ontology class: {}".format(ontology_class[0]))
                    classes.append(ontology_class[0])

            class_count = Counter(classes)
            #print(class_count)
            #print(len(no_defined_class))
        class_count = Counter(classes)
        print('total class count: {}'.format(class_count))
        print('total questions: {}'.format(total_q))
        return class_count

def write_dbpedia_relations():
    all_relations_with_label = {}
    uri_relations = []
    write_to_file = open("../data/all_DBpedia_relations.txt", "w+")

    with open('../data/dbontologyindex1.json') as f:
        for line in f:
            ln = line.strip()
            if "mergedLabel" in ln:
                uri = ln[ln.find("uri")+len("uri")+3:ln.find("urianalyzed")-3]
                uri_rel = uri[uri.rfind("/")+1:]
                label = ln[ln.rfind("mergedLabel")+len("mergedLabel")+3:ln.find("uri")-3]

                if uri_rel in uri_relations:
                    if label not in all_relations_with_label[uri_rel]:
                        all_relations_with_label[uri_rel].append(label)
                else:
                    uri_relations.append(uri_rel)
                    all_relations_with_label[uri_rel] = []
                    all_relations_with_label[uri_rel].append(label)


    with open('../data/all_relations_with_label.json', 'w') as outfile:
        json.dump([all_relations_with_label], outfile)
    write_to_file.close()


def find_similar_relations_using_thesaurus(word):
    w = Word(word)
    return w.synonyms()


# Returns a dictionary where key: value == relation: list_of_labels
def load_DBpedia_relations():
    with open('../data/all_relations_with_label.json') as f:
        relations = json.load(f)
        relations = relations[0]
        return relations

def build_train_test_data():
    # load dataset

    # load all dbpedia relations
    relations = load_DBpedia_relations()

    # get top 5 relations from dbpedia relations (extracted from dbpedia uri)

    # get top 5 relations related to the entity from dbpedia relations which are extracted from dbpedia relations

    # make a set( 1 exact relation + 5 similar relations from dbpedia + 5  similar relations from dbpedia for that entity

    pass


def unique_finder():
    #TEST
    print(textdistance.cosine("father of","uncle"))
    print(textdistance.cosine("father", "mother"))
    a = [2,4,6,2,5,7,2.3,12,31.39,3]
    a = np.array(a)
    print(np.argmax(a))
    print(a[np.argmax(a)])


#count_ontology_classes()
#write_dbpedia_relations()
#find_similar_relations_using_thesaurus()

#unique_finder()
load_DBpedia_relations()