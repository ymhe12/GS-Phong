DATADIR='your_dataset_path'
OUTPUT='output/your_data_name'

# For synthetic dataset
python train.py -s $DATADIR -m $OUTPUT --eval --port 6020
python train_meta.py -s $DATADIR -m $OUTPUT --eval --port 6020

# For realworld dataset
python train.py -s $DATADIR -m $OUTPUT --eval --port 6020 --realworld
python train_meta.py -s $DATADIR -m $OUTPUT --eval --port 6020 --realworld

# Evaluation
python render.py -m $OUTPUT
python metrics.py -m $OUTPUT