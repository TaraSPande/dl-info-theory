#!/bin/bash
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 6   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n6   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 17   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n17   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 51   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n51   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 148   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n148   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 431   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n431   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 1254   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n1254   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 3649   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n3649   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 10622   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n10622   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 30919   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n30919   --max-eval-samples 100
python main.py train   --task regression   --model transformer   --csv data/task_1_regression.csv   --features MolWt,MolLogP,TPSA,BertzCT,NumValenceElectrons   --label homolumogap   --max-train-samples 90000   --attn-self-enc vanilla
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n90000   --max-eval-samples 100