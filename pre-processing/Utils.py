import json
from thesaurus import Word
from collections import Counter

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
    all_rel =[]
    write_to_file = open("../data/all_DBpedia_relations.txt", "w+")
    with open('../data/dbontologyindex1.json') as f:
        for line in f:
            ln = line.strip()
            if "mergedLabel" in ln:
                relation = ln[ln.rfind("mergedLabel")+len("mergedLabel")+3:ln.find("uri")-3]
                if relation not in all_rel and len(relation)>0:
                    all_rel.append(relation.strip())
                    write_to_file.write(relation.strip()+"\n")
    write_to_file.close()


def find_similar_relations_using_thesaurus(word):
    w = Word(word)
    return w.synonyms()



def build_dataset():
    # get top 5 relations

    # get top 5 relations related to the entity

    pass


#count_ontology_classes()
#write_dbpedia_relations()
#find_similar_relations_using_thesaurus()
