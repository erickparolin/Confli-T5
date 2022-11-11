from transformers import pipeline
import numpy as np
import mlconjug3
import spacy
import torch
import pandas as pd
import re
import json
import csv


class nlg:
    
    def __init__(self, tokenizer, model, CAMEO_repo):
        self.nlp = spacy.load("en_core_web_sm")
        self.default_conjugator = mlconjug3.Conjugator(language='en')
        self.tokenizer = tokenizer
        self.model = model
        self.nationality2ctry = CAMEO_repo["nationality2ctry"]
        self.code2agents = CAMEO_repo["code2agents"]
        self.distr_events_actors = CAMEO_repo["distr_events_actors"] 
        self.fullcode2description = CAMEO_repo["fullcode2description"]
        self.patterns_dict = CAMEO_repo["patterns_dict"]
        self.synsets = CAMEO_repo["synsets"]
        
        
    
    def fill_the_blanks(self, sentence, p, temp1, sample_size = 1):
        blanks = sentence.count("<extra_id_")
        input_ids = self.tokenizer(sentence, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids.cuda(),do_sample=True, top_p=p,num_return_sequences=sample_size, temperature=temp1)
        raw_text = self.tokenizer.batch_decode(output_ids)

        all_sentences = []
        for s in raw_text:
            aux_sentence = sentence
            for _id in range(blanks):
                extra_id = "<extra_id_%s>"%(_id)
                extracted_t = re.search(extra_id+"([^<]+)",s)
                if extracted_t is not None:
                    extracted_t = extracted_t.group(1)
                else:
                    extracted_t = ""
                aux_sentence = aux_sentence.replace(extra_id, extracted_t).replace("  ", " ")


            all_sentences.append(aux_sentence)

        return all_sentences
    
    
    def assign_blanks(self, pattern, start_token):

        seq = 0
        out_pattern = ""
        do_blank = False

        parsed = self.nlp(pattern.lower().replace("<s>", "sourcex").replace("<t>", "targety").replace("_", " ").replace("*", ""))
        compound = [x.lower().replace("*", "") for x in pattern.split() if "_" in x and not x.endswith("_")]

        if "_" in start_token:
            start_token = start_token.split("_")[0]

        for idx, token in enumerate(parsed):

            tok_to_add = str(token.text).lower().replace("sourcex", "<s>").replace("targety", "<t>").lower()

            # out_pattern = out_pattern + " " + str(token.text)
            out_pattern = out_pattern + " " + tok_to_add
            if tok_to_add.lower() == start_token.lower():
                do_blank = True


            if do_blank and (str(token.text) not in [".", "_"]) and (idx < len(parsed)-1):

                ###### Checking whether is compound
                comp = str(token.text) + "_" + str(parsed[idx+1].text)
                comps_list = [x for x in compound if comp in x]
                # print("comps_list: ", comps_list)
                if len(comps_list) > 0:
                    continue

                ###### Checking whether it is an adjective to next token
                elif (token.pos_ == "ADJ") and (token.head.text == parsed[idx+1].text):
                    continue

                ###### Checking whether it is a PART to next token
                elif (token.pos_ == "PART") and (token.head.text == parsed[idx+1].text):
                    continue

                ###### Checking whether it is a PART and next token is AUX
                elif (token.pos_ == "PART") and (parsed[idx+1].pos_ == "AUX") and (token.head.text == parsed[idx+1].head.text):
                    continue

                ###### Checking whether it is an AUX to next token
                elif (token.pos_ == "AUX") and (token.head.text == parsed[idx+1].text):
                    continue

                ###### Checking whether it is a compound noun
                elif (token.dep_ == "compound") and (token.head.text == parsed[idx+1].text) and (parsed[idx+1].text != "targety"):
                    continue

                elif (parsed[idx+1].dep_ in ["prep", "pobj"]) and (parsed[idx+1].head.text == token.text):
                    continue

                elif (parsed[idx+1].dep_ == "dobj") and (parsed[idx+1].head.text == token.text):
                    continue

                else:
                    out_pattern = out_pattern + " " + "<extra_id_"+str(seq)+">"
                    seq = seq + 1

        out_pattern = out_pattern[:-1]

        return out_pattern, parsed

    
    def sample_pattern(self, all_patterns, code_distr):

        action_code = np.random.choice(a=list(all_patterns.keys()), p=code_distr)
        raw_pat, raw_code = np.random.choice(a=all_patterns[action_code]).split("_id_")

        # Remove it later, once we fix the issues/ambiguities with "%" sign.
        while "%" in raw_pat:
            raw_pat, raw_code = np.random.choice(a=all_patterns[action_code]).split("_id_")


        # Evaluating case by case, depending on whether the pattern contains %, $ or +
        final_pat = raw_pat
        if "%" in raw_pat:
            final_pat = final_pat.replace("%", "<S> and <T>")
        if "$" in raw_pat:
            final_pat = final_pat.replace("$", "<S>")
        if "+" in raw_pat:
            final_pat = final_pat.replace("+", "<T>")
        if "%" not in raw_pat and "$" not in raw_pat and "+" not in raw_pat:
            final_pat = "<S> "+final_pat+" <T>"


        # Fixing some cases where we have Source but not Target
        if "<S>" not in final_pat:
            if final_pat.startswith("<T>"):
                final_pat = final_pat + " <S>"
            else:
                final_pat = "<S> " + final_pat

        if "<T>" not in final_pat:
            if final_pat.startswith("<S>"):
                final_pat = final_pat + " <T>"
            else:
                final_pat = "<T> " + final_pat

        final_pat = final_pat + "."


        # Parsing and replacing synsets
        while "&" in final_pat:
            syn = "&"+final_pat.split("&")[1].split()[0].split("_")[0].replace(".", "")
            sampled_syn = np.random.choice(a=self.synsets[syn])
            final_pat = final_pat.replace(syn, sampled_syn)


        # Changing the verbal tense of the chosen pattern
        if raw_pat.startswith("*") or raw_pat.startswith("+ &AUXVERB1 *") or raw_pat.startswith("+ *") or raw_pat.startswith("$ *") or raw_pat.startswith("+ WAS *") or raw_pat.startswith("+ WERE *"):

            # (i) identifying the root verb
            root_verb = re.search('\*(.*)\*', final_pat).group(1)

            if "_" not in root_verb:
                root_verb_ = self.default_conjugator.conjugate(root_verb.lower()).conjug_info['indicative']['indicative past tense']['2s']
                final_pat = final_pat.replace(root_verb, root_verb_)
            else:
                root_verb_ = root_verb.replace("*", "").lower()

        else:
            root_verb_ = final_pat.split()[0]

        # Assigning blanks along the patterns to be filled out.
        final_pat, parse = self.assign_blanks(final_pat, root_verb_)


        final_pat = final_pat.replace("*", "").lower()

        return final_pat, raw_pat, raw_code, int(action_code), parse

  
    def get_actors(self, text, raw_code, raw_pat, description):

        rootcode = str(int(raw_code[:2]))
        S = None
        T = None

        # Selecting the Source
        while S is None:
            if ("IN $" in raw_pat) or ("VISITED +" in raw_pat):
                S = np.random.choice(a=list(self.nationality2ctry.values()))
            else:
                src_code = np.random.choice(a=list(self.distr_events_actors[rootcode]["Sources"].keys()), p=list(self.distr_events_actors[rootcode]["Sources"].values()))
                if src_code.replace("~", "") not in self.code2agents:
                    continue
                elif src_code == "CTRY":
                    S = np.random.choice(a=list(self.nationality2ctry.values()))
                elif src_code.startswith("~"):
                    S = np.random.choice(a=self.code2agents[src_code[1:]])
                    S = S.replace("_", " ").replace("  ", " ").strip().title()
                    nat = np.random.choice(a=list(self.nationality2ctry.keys()))
                    S = nat + " " + S
                else:
                    S = np.random.choice(a=self.code2agents[src_code])
                    S = S.replace("_", " ").replace("  ", " ").strip().title()

        # Selecting the Target 
        while T is None:
            if ("IN +" in raw_pat) or (raw_pat.endswith("IN") and "+" not in raw_pat) or ("VISITED +" in raw_pat) or (raw_pat.endswith("VISITED") and "+" not in raw_pat):
                T = np.random.choice(a=list(self.nationality2ctry.values()))
            else:
                tgt_code = np.random.choice(a=list(self.distr_events_actors[rootcode]["Targets"].keys()), p=list(self.distr_events_actors[rootcode]["Targets"].values()))
                if tgt_code.replace("~", "") not in self.code2agents:
                    continue
                elif tgt_code == "CTRY":
                    T = np.random.choice(a=list(self.nationality2ctry.values()))
                elif tgt_code.startswith("~"):
                    T = np.random.choice(a=self.code2agents[tgt_code[1:]])
                    T = T.replace("_", " ").replace("  ", " ").strip().title()
                    nat = np.random.choice(a=list(self.nationality2ctry.keys()))
                    T = nat + " " + T
                else:
                    T = np.random.choice(a=self.code2agents[tgt_code])
                    T = T.replace("_", " ").replace("  ", " ").strip().title()

        return text.replace("<s>", S).replace("<t>", T), description.replace("<s>", S).replace("<t>", T), S, T

    
    def generate(self, p, temp1, outfile="data/D_tilde.json", n_sentences=1000, code_distr=None, actors_distr=None):

        fabricated_training_data = []
        root2penta = {'01': 0, 
                      '02': 0, 
                      '03':1, 
                      '04':1, 
                      '05':1, 
                      '06':2, 
                      '07':2, 
                      '08':2, 
                      '09': 3, 
                      '10':3, 
                      '11':3, 
                      '12':3, 
                      '13':3, 
                      '16':3, 
                      '14':4, 
                      '15':4, 
                      '17': 4, 
                      '18': 4, 
                      '19': 4, 
                      '20':4}

        ## Universe of verbal patterns "all_patterns" will be on root codes level by default
        if code_distr is None or len(code_distr) == 20:
            all_patterns = {int(root):[] for root in range(1,21)}
            for code in self.patterns_dict.keys():
                new_pats = [pat+"_id_"+code for pat in self.patterns_dict[code]]
                all_patterns[int(code[:2])] = all_patterns[int(code[:2])] + new_pats

            if code_distr is None:
                 code_distr = [.05]*20

                    
        elif len(code_distr) == 5:
            all_patterns = {penta:[] for penta in range(5)}
            for code in self.patterns_dict.keys():
                new_pats = [pat+"_id_"+code for pat in self.patterns_dict[code]]
                penta = root2penta[code[:2]]
                all_patterns[penta] = all_patterns[penta] + new_pats


        else:
            print("Wrong dimensionality for code_distr. Please enter 5 or 20 categories only.")
            return False


        # Generating n_sentences fabricated/syntectic sentences
        for i in range(n_sentences):

            # try:

            # Choosing the verbal pattern and root_code based on actions_distr
            pattern, raw_pat, raw_code, grouped_code, parse = self.sample_pattern(all_patterns, code_distr)


            text, description, S, T = self.get_actors(pattern, raw_code, raw_pat, self.fullcode2description[raw_code])
            full_sentence = self.fill_the_blanks(description + " "+ text, p, temp1)[0]
            train_sentence = full_sentence.replace(description, "")

            fabricated_training_data.append({"train_sentence": str(train_sentence), "description": description, "raw_pat": raw_pat, "raw_code": raw_code, "Source": S, "Target": T})

            
        with open(outfile, "w") as outF:
            json.dump(fabricated_training_data, outF)
            
            
            
    
class nli:
    
    def __init__(self, nli_model="facebook/bart-large-mnli", device=None):
        
        if device is not None:
            self.nli_model = pipeline("zero-shot-classification", model=nli_model, device=device)
        else:
            self.nli_model = pipeline("zero-shot-classification", model=nli_model)
            
        
    def nli_parse(self, D_full_file="data/D_tilde.json", materialize_nli=True):
        
        D_full_nli = []
        
        with open(D_full_file) as json_file:
            D_full = json.load(json_file)
            
        
        for sent in D_full:
            generated_text = sent['train_sentence'].replace(" ing ", "ing ").replace(" s ", "s ").replace("suspen ", "suspend")
            description = sent['description']

            zs = self.nli_model(generated_text, description, multi_label=True)
            sent["score"] = zs["scores"][0]
            sent["rootcode"] = sent["raw_code"][:2]

            D_full_nli.append(sent)
            

        if materialize_nli:
            out_fileName = D_full_file.split(".")[0]+"_nli."+D_full_file.split(".")[1]
            with open(out_fileName, "w") as json_file:
                json.dump(D_full_nli, json_file)

        return D_full_nli
        
        
def topQ_sampling(D_full_nli, CAMEO2labels, CAMEO2distr, D_file, D, N, q, temp2):

    samples_to_generate = N - len(D)
    
    D_full_nli_df = pd.DataFrame(D_full_nli)

    for code, prob in CAMEO2distr.items():
        y = CAMEO2labels[code]
        samples_to_generate_code = int(samples_to_generate * prob)
        
        # Filtering out samples with nli < q
        cut_sample = D_full_nli_df.loc[(D_full_nli_df['rootcode'] == str(code).zfill(2)) & (D_full_nli_df['score'] >= q)]
        sents = [sent.replace(" ing ", "ing ").replace(" s ", "s ") for sent in cut_sample['train_sentence'].tolist()]
        
        # Re-computing the probability 
        scores = cut_sample['score'].tolist()
        probs = (np.exp(np.array(scores)/temp2)) / sum((np.exp(np.array(scores)/temp2)))
        
        if samples_to_generate_code > len(sents):
            random_sents = [[sent, y] for sent in np.random.choice(a=sents, size=samples_to_generate_code, replace=True, p=probs)]
        else:
            random_sents = [[sent, y] for sent in np.random.choice(a=sents, size=samples_to_generate_code, replace=False, p=probs)]

        D = D + random_sents

    with open(D_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(D)

        
        
        
        