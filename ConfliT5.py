from ct5_model import nlg, nli, topQ_sampling
from cameo import import_CAMEO
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import pandas as pd
import numpy as np
import argparse
import warnings
import torch
import os
import re
import json
import csv

warnings.simplefilter("ignore", UserWarning)


def main():
    
    parser = argparse.ArgumentParser()

    ## Main parameters
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='Whether to use GPU acceleration.')
    # parser.add_argument('--do_nlg', type=bool, default=True, help='Run NLG step (set false if D_tilde.json exists.)')
    # parser.add_argument('--do_nli', type=bool, default=True, help='Run NLI step (set false if D_tilde_nli.json exists.)')
    
    parser.add_argument('--no_nlg', default=False, action="store_true", help='Run NLG step (set it if D_tilde.json exists.)')
    parser.add_argument('--no_nli', default=False, action="store_true", help='Run NLI step (set it if D_tilde_nli.json exists.)')
    parser.add_argument('--device', type=int, default=0, help="GPU #")
    
    parser.add_argument('--no_materialize_nli', default=False, action="store_true", help='Materialize data after NLI parsing?')
    
    parser.add_argument('--nlg_model', type=str, default="t5-large", help="NLG model.")
    parser.add_argument('--nli_model', type=str, default="facebook/bart-large-mnli", help="NLI model.")
    parser.add_argument('--CAMEO2labels_file', type=str, default="data/CAMEO2labels.json", help="CAMEO2labels dictionary.")
    parser.add_argument('--CAMEO2distr_file', type=str, default="data/CAMEO2distr.json", help="CAMEO2distr dictionary.")
    parser.add_argument('--D_full_file', type=str, default="data/D_tilde.json", help="Path to D_tilde file (existing or to be created).")
    parser.add_argument('--D_file', type=str, default="data/D.json", help="Path to D file (to be created).")
    parser.add_argument('--D_full_size', type=int, default=10000, help="Size of full synthetic data D_tilde (before top-q sampling).")
    parser.add_argument('--code_distr', type=str, default=None, help="Distribution of CAMEO codes to generate D_full.")
    parser.add_argument('--actors_distr', type=str, default=None, help="Distribution of CAMEO actors to generate D_full.")
    parser.add_argument('--N', type=int, default=3000, help="Size of final training data D (after top-q sampling).")
    parser.add_argument('--p', type=float, default=0.9, help="Hyperparameter p for nucleus sampling.")
    parser.add_argument('--temp1', type=float, default=0.95, help="Hyperparameter temp1 for nucleus sampling.")
    parser.add_argument('--q', type=float, default=0.975, help="Hyperparameter q for top-q sampling.")
    parser.add_argument('--temp2', type=float, default=0.9, help="Hyperparameter temp2 for top-q sampling.")
    parser.add_argument('--annotated_path', type=str, default="data/cameo_annotated_050.csv", help="Pre-existing labeled data.")
    
    args = parser.parse_args()


    # Importing CAMEO dictionaries
    CAMEO_repo = import_CAMEO()

    # Importing CAMEO2labels and CAMEO2distr dictionaries
    with open(args.CAMEO2labels_file) as json_file:
        CAMEO2labels = json.load(json_file)

    with open(args.CAMEO2distr_file) as json_file:
        CAMEO2distr = json.load(json_file)

    # NLG model
    if args.no_nlg:
        pass
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.nlg_model)
        model = T5ForConditionalGeneration.from_pretrained(args.nlg_model)
        if args.cuda:
            model.cuda()
            
        model.eval()

        # Generator Object
        NLG = nlg(tokenizer, model, CAMEO_repo)

        ## Generating the synthetic samples using NLG
        NLG.generate(p=args.p, 
                     temp1=args.temp1, 
                     outfile=args.D_full_file, 
                     n_sentences=args.D_full_size, 
                     code_distr=args.code_distr, 
                     actors_distr=args.actors_distr
                    )

        # Empty GPU (NLG model)
        if args.cuda:
            torch.cuda.empty_cache()
        
        print(">> NLG step concluded...")
        
    # NLI model 
    if args.no_nli:
        D_full_file_nli = args.D_full_file.split(".")[0]+"_nli."+args.D_full_file.split(".")[1]
        with open(D_full_file_nli) as json_file:
            D_full_nli = json.load(json_file)
            
    else:
        NLI = nli(nli_model=args.nli_model, device=args.device)

        ## NLI parsing the synthetic samples
        D_full_nli = NLI.nli_parse(D_full_file=args.D_full_file, 
                                   materialize_nli = not args.no_materialize_nli
                                  )
        
        print(">> NLI step concluded...")
    
    ## Importing annotated data (capital lambda in pseudo-code)
    D = []
    if args.annotated_path is not None:
        with open(args.annotated_path) as fd:
            rd = csv.reader(fd, delimiter = "\t")
            for row in rd:
                D.append([row[0], int(row[1])])    

    # Top-q sampling to create the synthetic training data D and materialize it in D_file
    topQ_sampling(D_full_nli=D_full_nli, 
                  CAMEO2labels=CAMEO2labels, 
                  CAMEO2distr=CAMEO2distr, 
                  D_file=args.D_file, 
                  D=D, 
                  N=args.N, 
                  q=args.q, 
                  temp2=args.temp2
                 )

    


if __name__ == "__main__":
    main()