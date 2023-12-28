# python ../src/main.py GRU \
#         --epochs 46 \
#         --batch_size 128 \
#         --lr 0.0001 \
#         --save_path ../ckpt/GRU125MES.pt \
#         --save_model \
#         --trace_time \
#         --encoder_layers 10 \
#         --decoder_layers 10 \
#         --embedding_size 512 \
#         --hidden_size 1024 \
#         --direction 1 \

python ../src/main.py GRU \
        --epochs 43 \
        --batch_size 128 \
        --lr 0.0001 \
        --save_path ../ckpt/GRU100MES.pt \
        --save_model \
        --trace_time \
        --encoder_layers 8 \
        --decoder_layers 8 \
        --embedding_size 512 \
        --hidden_size 1024 \
        --valid_loss \
        --direction 1 \

python ../src/main.py GRU \
        --epochs 25 \
        --batch_size 128 \
        --lr 0.0001 \
        --save_path ../ckpt/GRU75MES.pt \
        --save_model \
        --trace_time \
        --encoder_layers 6 \
        --decoder_layers 6 \
        --embedding_size 512 \
        --hidden_size 1024 \
        --valid_loss \
        --direction 1 \