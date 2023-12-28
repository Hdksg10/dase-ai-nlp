python ../src/main.py Transformer \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.0001 \
        --load_path ../ckpt/Transformer53MES.ptTransformerEncoderDecoder.pt \
        --load_model \
        --trace_time \
        --encoder_layers 6 \
        --decoder_layers 6 \
        --embedding_size 512 \
        --hidden_size 2048 \
        --num_heads 8 \

python ../src/main.py Transformer \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.0001 \
        --load_path ../ckpt/Transformer68MES.ptTransformerEncoderDecoder.pt \
        --load_model \
        --trace_time \
        --encoder_layers 8 \
        --decoder_layers 8 \
        --embedding_size 512 \
        --hidden_size 2048 \
        --num_heads 8 \

python ../src/main.py GRU \
        --epochs 43 \
        --batch_size 128 \
        --lr 0.0001 \
        --load_path ../ckpt/GRU100MES.ptGRUEncoderDecoder.pt \
        --load_model \
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
        --load_path ../ckpt/GRU75MES.ptGRUEncoderDecoder.pt \
        --load_model \
        --trace_time \
        --encoder_layers 6 \
        --decoder_layers 6 \
        --embedding_size 512 \
        --hidden_size 1024 \
        --direction 1 \