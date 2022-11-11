# Confli-T5: An AutoPrompt Pipeline for Conflict Related Text Augmentation

This repository contains the code for the paper "Confli-T5: An AutoPrompt Pipeline for Conflict Related Text Augmentation" (IEEE Big Data 2022).

## Prerequisites
This code was written in Python 3.7 and requires the following packages:

    pytorch==1.8.0
    numpy==1.19.5
    pandas==1.1.3
    transformers==4.11.3
    mlconjug3==3.7.20
    spacy==3.1.1

## Augmentating data
For data augmentation using Confli-T5, simply run the code ConfliT5.py, as described below:
	
```	
	python ConfliT5.py --cuda \
	--device 0 \
	--nlg_model t5-large \
	--nli_model facebook/bart-large-mnli \
	--CAMEO2labels_file ./data/CAMEO2labels.json \
	--CAMEO2distr_file ./data/CAMEO2distr.json \
	--D_full_file ./data/D_tilde.json \
	--D_file ./data/D.json \
	--overwrite_output_dir \
	--D_full_size 10000 \
	--code_distr YOUR_VALID_FILE \
	--N 3000\ 
	--p 0.9 \
	--temp1 0.95 \
	--q 0.975 \
	--temp2 0.9 \
	--annotated_path ./data/cameo_annotated_050.csv \
```



## Citation

If you find this repo useful in your research, please cite:

    @inproceedings{parolin2022conf,
      title={Confli-T5: An AutoPrompt Pipeline for Conflict Related Text Augmentation},
      author={Parolin, Erick Skorupa and Hu, Yibo and Khan, Latifur and Osorio, Javier and Brandt, Patrick T and D’Orazio, Vito},
      booktitle={2022 IEEE International Conference on Big Data (Big Data)},
      year={2022},
      organization={IEEE}
    }
