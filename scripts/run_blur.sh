$1


python main.py 0 --arch $1 -p blur --prefix blur --lr 0.001 --momentum 0.7 --num_epochs 25 --decay_factor 3 --batch_size 15 --eval_freq 1
python main.py 1 --arch $1 -p blur --prefix blur --lr 0.001 --momentum 0.7 --num_epochs 25 --decay_factor 3 --batch_size 15 --eval_freq 1
python main.py 2 --arch $1 -p blur --prefix blur --lr 0.001 --momentum 0.7 --num_epochs 25 --decay_factor 3 --batch_size 15 --eval_freq 1
python main.py 3 --arch $1 -p blur --prefix blur --lr 0.001 --momentum 0.7 --num_epochs 25 --decay_factor 3 --batch_size 15 --eval_freq 1
python main.py 4 --arch $1 -p blur --prefix blur --lr 0.001 --momentum 0.7 --num_epochs 25 --decay_factor 3 --batch_size 15 --eval_freq 1

