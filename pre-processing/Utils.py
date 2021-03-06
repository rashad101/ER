import json
from thesaurus import Word
from collections import Counter
import textdistance
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import nltk
import re
import csv
import heapq
from SPARQLWrapper import SPARQLWrapper, JSON



#returns a list of triple-list
def load_data():
    with open('../data/FullyAnnotated_LCQuAD_new.json') as f:
        data = json.load(f)
        triple_list = []
        for i in range(len(data)):
            for dict in data[i]:
                subject_uri = dict["subject"]
                subject = subject_uri[subject_uri.rfind("/")+1:]

                object_uri = dict["object"]
                object_ = object_uri[object_uri.rfind("/")+1:]

                predicate_uri = dict["predicate"]
                predicate = predicate_uri[predicate_uri.rfind("/")+1:]

                triple_list.append([subject,predicate,object_])
        return triple_list



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

    #print(len(all_relations_with_label))
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
        print(len(relations))
        return relations

def build_data():

    # load dataset (triples)
    data = load_data()
    total = len(data)

    # load all dbpedia relations
    relations = load_DBpedia_relations()
    relations = list(relations.keys())
    er_n_list = []
    j = 0
    for i in range(len(data)):
        sub = data[i][0]
        pred = data[i][1]
        obj = data[i][2]

        relation_set = set()

        # get top 5 relations from dbpedia relations (extracted from dbpedia uri)
        similarity = []
        for rel in relations:
            if rel!=pred:
                similarity.append({"rel":rel, "similarity": textdistance.cosine(rel,pred)})

        top5_rel = heapq.nlargest(5, similarity, key=lambda s: s['similarity'])

        relation_set.add(pred)

        for rel in top5_rel:
            relation_set.add(rel['rel'])


        # get top 5 relations related to the entity from dbpedia relations which are extracted from dbpedia relations
        relations_for_entity = get_relations(sub)

        similarity = []
        for rel in relations_for_entity:
            if rel!=pred:
                similarity.append({"rel":rel, "similarity": textdistance.cosine(rel,pred)})

        top5_er = heapq.nlargest(len(similarity),similarity, key= lambda s: s['similarity'])


        taken =0
        blacklist = ["wikiPageDisambiguates","wikiPageWikiLink","wikiPageRedirects"]
        for rel in top5_er:
            if taken<5 and rel['rel'] not in blacklist:
                relation_set.add(rel['rel'])
                taken+=1


        # make a set ( 1 exact relation + 5 similar relations from dbpedia + 5  similar relations from dbpedia for that entity)
        d = {}
        d["subject"] = sub
        d["predicate"] = pred
        d["list_of_relations"] = list(relation_set)

        er_n_list.append(d)
        j+=1
        if j%50==0:
            percent = (j/total)*100.0
            print("done %.2f%%"%percent)

    with open('../data/ER_n_relList_v2.json', 'w') as outfile:
        json.dump([er_n_list], outfile, indent=4)


def get_relations(entity):

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("SELECT distinct ?rel where { ?obj ?rel <http://dbpedia.org/resource/"+entity+">}")
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    results = results['results']['bindings']
    link_set = set()
    for data in results:
        rel_uri = data['rel']['value']
        rel = rel_uri[rel_uri.rfind("/")+1:]
        link_set.add(rel)
    return  list(link_set)


def remove_punct(sentence, keep_apostrophe = False):
    sentence = sentence.strip()
    if keep_apostrophe:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    else :
        PATTERN = r'[^a-zA-Z0-9]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    return(filtered_sentence)



def build_lcquad_test():
    all_er = {}

    e=0
    r=0
    with open('../data/FullyAnnotated_LCQuAD5000.json') as f:
        data = json.load(f)
        for d in data:
            e_mapping = d["entity mapping"]
            for ent in e_mapping:
                all_er[ent["label"]]="E"
                e+=1
            pd_mapping = d["predicate mapping"]
            for pd in pd_mapping:
                if "label" in pd:
                    all_er[pd["label"]]="R"
                    r+=1

        print("Total entities: {}".format(e))
        print("Total relations: {}".format(r))

    rows = []
    for k,v in all_er.items():
        rows.append([k,v])

    to_write = open("../data/Test_LC_QuAD.csv","w")
    data_writer = csv.writer(to_write)
    data_writer.writerows(rows)


#count_ontology_classes()
#write_dbpedia_relations()
build_data()
#get_relations("The_Invisible_(band)")
#build_lcquad_test()
#load_DBpedia_relations()

#The_Invisible_(band)