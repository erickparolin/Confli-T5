import json

def import_CAMEO(cameo_dict="data/allDicts.json", countries_dict="data/countries.json", distr_events='data/distr_events_actors.json', codebook="data/cameo_descriptions.txt", action_dicts = "data/CAMEO.verbpatterns.150430.txt"):

    all_CAMEO = {}
    
    ############################################################
    ####      Importing Actors Dictionaries    #################
    ############################################################
    with open(cameo_dict) as json_file:
        allDicts = json.load(json_file)

    with open(countries_dict) as json_file:
        countries = json.load(json_file)
    nationality2ctry = {el['nationality']: el['en_short_name'] for el in countries}
    
    all_CAMEO["allDicts"] = allDicts
    all_CAMEO["countries"] = countries
    all_CAMEO["nationality2ctry"] = nationality2ctry
    

    ####################################################################################################################
    ####      Selecting Actors to be used for Generation process (Agents and International entities)   #################
    ####################################################################################################################
    code2agents = {}
    for agent, rules in allDicts['agents2codes'].items():
        for code, zero in rules.items():
            if code.replace("~", "") not in code2agents:
                code2agents[code.replace("~", "")] = [agent]
            else:
                code2agents[code.replace("~", "")].append(agent)

    international_ents = []
    for agent, rules in allDicts['international2codes'].items():
        for code, zero in rules.items():
            if code.replace("~", "") not in code2agents:
                code2agents[code.replace("~", "")] = [agent]
            else:
                code2agents[code.replace("~", "")].append(agent)

            if code not in international_ents:
                international_ents.append(code.replace("~", ""))

                
    all_CAMEO["code2agents"] = code2agents
    all_CAMEO["international_ents"] = international_ents

    ################################################################################################################
    ####      Importing Distribution of Actor Codes by RootCode (from real-world processed news)   #################
    ################################################################################################################
    with open(distr_events) as json_file:
        distr_events_actors = json.load(json_file)    


    all_CAMEO["distr_events_actors"] = distr_events_actors
    
    ########################################################
    ####     Reading action-patterns dictionary     ########
    ########################################################
    codesFile = open(codebook, 'r', encoding='utf-8')
    codesFile = codesFile.readlines()

    fullcode2description = {}
    for line in codesFile:
        fullcode = line.split(":")[0]
        description = line.split(":")[1].replace("\n", "").lower().strip()
        fullcode2description[fullcode] = description

    all_CAMEO["fullcode2description"] = fullcode2description
        
    ##########################################################
    ####     Importing Actions Dictionaries P1     ###########
    ##########################################################

    ## READING ACTION-PATTERNS DICTIONARY
    actionsDictFile = open(action_dicts, 'r', encoding='utf-8')
    actionsFile = actionsDictFile.readlines()


    patterns_dict = {i:[] for i in fullcode2description.keys()}
    synsets = {}
    patterns_line = False
    S_patterns = 0
    T_patterns = 0
    no_ST_patterns = 0
    compound_patterns = 0
    ST_patterns = 0

    for line_num, line in enumerate(actionsFile):

        if line.startswith("####### VERB PATTERNS ####### "):
            patterns_line= True

        elif line.startswith("# "):
            continue

        else:

            if not patterns_line:
                ## Then this is a synset...

                if line.startswith("&"):
                    key = line.replace("\n", "")
                    if key not in synsets:
                        synsets[key] = []

                elif line.startswith("+"):
                    syn = line.replace("+", "").replace("\n", "")
                    synsets[key].append(syn)

            else:
                ## Then this is a verbal pattern...

                if line.startswith("- ") and "# " in line and "[" in line and "[---]" not in line and "didn't work" not in line and"^" not in line: # Maybe review  the ^ cases later

                    # Extracting action pattern
                    verb = line.replace("\n", "").split("#")[1].split("<")[0].split(",")[0].strip()
                    pattern = line.replace("\n", "").split("#")[0].split("[")[0].strip()
                    pattern = pattern.replace("-", "").replace("*", "*"+verb+"*").strip()

                    # Extracting the action code associated to action pattern
                    if "[:" in line:
                        fullcode = line.split("[:")[1].split("]")[0].split(":")[0]
                    else:
                        fullcode = line.split("#")[0].split("[")[1].split("]")[0]
                        if ":" in fullcode:
                            if fullcode.split(":")[0][:2] == fullcode.split(":")[1][:2]:
                                fullcode = fullcode.split(":")[0]
                            else:
                                continue

                    if fullcode in fullcode2description:

                        fullcode = fullcode.replace("!", "")
                        rootcode = fullcode[:2]
                        patterns_dict[fullcode].append(pattern)


                        if ("%" in pattern):
                            compound_patterns=compound_patterns+1
                        if ("$" in pattern and "+" in pattern):
                            ST_patterns = ST_patterns + 1
                        elif ("$" in pattern): # or ("%" in pattern):
                            S_patterns=S_patterns+1
                        elif ("+" in pattern): # or ("%" in pattern):
                            T_patterns=T_patterns+1
                        if "$" not in pattern and "+" not in pattern and "%" not in pattern:
                            no_ST_patterns = no_ST_patterns + 1

    print("Patterns containing compound entities: ", compound_patterns)
    print("Patterns containing both source and target: ", ST_patterns)
    print("Patterns containing source only: ", S_patterns)
    print("Patterns containing target only: ", T_patterns)
    print("Patterns containing neither source nor target: ", no_ST_patterns)
    
    all_CAMEO["patterns_dict"] = patterns_dict
    all_CAMEO["synsets"] = synsets
    
    return all_CAMEO