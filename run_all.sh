#!/bin/bash

echo "===================================================="
echo " Activating LaneATT environment (manual) "
echo "===================================================="

# deactivate conda quietly
conda deactivate >/dev/null 2>&1

# cd into LaneATT project directory
cd /home/bozhen2/ComputerVision/lanetatt-classproject/external/LaneATT/ || exit

# activate poetry venv
source /home/bozhen2/.cache/pypoetry/virtualenvs/laneatt-classproj-TAaFv3Zi-py3.10/bin/activate

# print confirmation message
echo "[Env] Poetry virtual environment activated successfully!"
echo "[Env] Python used: $(which python)"
echo "[Env] Python version: $(python --version)"
echo "===================================================="
echo ""

echo "===================================================="
echo " TRAINING: ResNet18 on CULane "
echo "===================================================="

python main.py train --exp_name culane_resnet18 --cfg cfgs/laneatt_culane_resnet18.yml

echo "===================================================="
echo " FINISHED: ResNet18 "
echo "===================================================="
echo ""


echo "===================================================="
echo " TRAINING: ResNet34 on CULane "
echo "===================================================="

python main.py train --exp_name culane_resnet34 --cfg cfgs/laneatt_culane_resnet34.yml

echo "===================================================="
echo " FINISHED: ResNet34 "
echo "===================================================="
echo ""


echo "===================================================="
echo " TRAINING: ResNet122 on CULane "
echo "===================================================="

python main.py train --exp_name culane_resnet122 --cfg cfgs/laneatt_culane_resnet122.yml

echo "===================================================="
echo " FINISHED: ResNet122 "
echo "===================================================="
echo ""

echo "===================================================="
echo " ALL TRAINING COMPLETE "
echo "===================================================="
