python ../src/main.py Transformer \
        --epochs 16 \
        --batch_size 128 \
        --lr 0.0001 \
        --trace_time \
        --encoder_layers 8 \
        --decoder_layers 8 \
        --embedding_size 512 \
        --hidden_size 2048 \
        --num_heads 8 \

python ../src/main.py Transformer \
        --epochs 16 \
        --batch_size 128 \
        --lr 0.0001 \
        --trace_time \
        --encoder_layers 6 \
        --decoder_layers 6 \
        --embedding_size 512 \
        --hidden_size 2048 \
        --num_heads 8 \

python ../src/main.py GRU \
        --epochs 43 \
        --batch_size 128 \
        --lr 0.0001 \
        --trace_time \
        --encoder_layers 8 \
        --decoder_layers 8 \
        --embedding_size 512 \
        --hidden_size 1024 \
        --direction 1 \

python ../src/main.py GRU \
        --epochs 25 \
        --batch_size 128 \
        --lr 0.0001 \
        --trace_time \
        --encoder_layers 6 \
        --decoder_layers 6 \
        --embedding_size 512 \
        --hidden_size 1024 \
        --direction 1 \