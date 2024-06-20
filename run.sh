DATADIR='your_dataset_path'
OUTPUT='output/your_data_name'
python train.py -s $DATADIR -m $OUTPUT --eval --port 6020
python train_meta.py -s $DATADIR -m $OUTPUT --eval --port 6020
python render.py -m $OUTPUT
python metrics.py -m $OUTPUT