import csv
import json
from operator import itemgetter

def get_top_6000k_entities():
    i=0
    e_dict={}  #{entity:edgecount}
    with open('../data/dbentityindex11.json') as f:
        for line in f.readlines():
            if "dbpediaLabel" in line:
                label = line[line.rfind("dbpediaLabel")+len("dbpediaLabel")+3:line.rfind("}}")-1]
                if "uri" in line[line.find("_id"):line.find("_score")]:
                    edge_count = line[line.find("edgecount") + len("edgecount") + 2:line.rfind("uri") - 2]
                else:
                    edge_count = line[line.find("edgecount")+len("edgecount")+2:line.find("uri")-2]


                e_dict[label] = int(edge_count)

            else:
                label = line[line.find("wikidataLabel") + len("wikidataLabel") + 3:line.rfind("uri") - 3]



                if line.rfind("edgecount")> (len(line)-25): #edgecount is found at the end of the line
                    edge_count = line[line.rfind("edgecount") + len("edgecount") + 2:line.rfind("}}")]
                else:
                    label = line[line.find("wikidataLabel") + len("wikidataLabel") + 3:line.find("edgecount")-3]
                    edge_count = line[line.rfind("edgecount") + len("edgecount") + 2:line.find("uri")-2]


                if len(edge_count)>0:
                    e_dict[label] = int(edge_count)
            i+=1

    print("Total entity found: {}".format(i))
    print("Extracted {} unique entities".format(len(e_dict.keys())))

    #sorting dictionary based on edge_count
    sorted_dict = dict(sorted(e_dict.items(), key=itemgetter(1) , reverse=True))

    #Keeping only top 6000k entities. storing them as follows: (3000k-Capitalized) + (3000k- lowercased)
    write_to_file = []
    i=1
    for k,v in sorted_dict.items():
        if i%2==0:
            write_to_file.append([k.capitalize(),v])
        else:
            write_to_file.append([k.lower(),v])
        i+=1
        if i>6000000:
           break

    print("Writing 6000k entities to file...")
    filename = open("../data/6000k_entites.csv","w",encoding="utf-8")
    writer = csv.writer(filename)
    writer.writerows(write_to_file)
    filename.close()
    print("Done!!")


def build_data():

    er_dict = {}

    # Get 6000k relations
    rel = set()
    with open("../data/relations.txt", "r", encoding="utf-8") as f:
        for line in f:
            rel.add(line.strip())
    f.close()
    for r in rel:
        er_dict[r]="R"

    print(len(er_dict.keys()))


#get_top_6000k_entities()
build_data()