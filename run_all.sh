#!/bin/bash
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 16   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n16   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 44   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n44   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 118   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n118   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 317   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n317   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 856   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n856   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 2309   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n2309   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 6226   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n6226   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 16792   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n16792   --max-eval-samples 100
python main.py train   --task classification   --model transformer   --csv data/task_1_classification.csv   --features V1,V2,V3,V4,V5   --label Y   --max-train-samples 45286   --attn-self-enc vanilla
python main.py eval   --run ./runs/classification-transformer-task_1_classification-enc6d512h8-vanilla_n45286   --max-eval-samples 100