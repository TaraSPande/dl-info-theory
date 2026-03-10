#!/bin/bash
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n6   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n15   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n38   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n94   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n236   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n591   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n1479   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n3703   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n9273   --max-eval-samples 100
python main.py eval   --run ./runs/regression-transformer-task_1_regression-enc6d512h8-vanilla_n23218   --max-eval-samples 100