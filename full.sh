# /bin/sh
./run.sh -f data/spanish.train.txt -c 5 -i 200 -g 2.0 -m spanish -T 
./run.sh -f data/spanish.train.test.txt -m spanish > data/result-spanish
cd data

./eval.sh spanish.train.txt result-spanish     
cd ..
