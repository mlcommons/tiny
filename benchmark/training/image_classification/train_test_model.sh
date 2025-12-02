# train and test the model

if [ ! -d "Logs" ]; then
    mkdir -p Logs
fi

python3 train.py > Logs/training_log.txt
python3 test.py > Logs/testing_log.txt