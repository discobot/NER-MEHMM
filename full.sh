# /bin/sh
./run.sh -f data/spanish.train.txt -c 1 -i 10000 -m spanish -T 
./run.sh -f data/spanish.train.test.txt -m spanish > data/result-spanish
cd data

./eval.sh spanish.train.txt result-spanish     
cd ..
