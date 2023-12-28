# python ../src/main.py Transformer \
#         --epochs 140 \
#         --batch_size 256 \
#         --lr 0.0001 \
#         --save_path ../ckpt/Transformer100M.pt \
#         --save_model \
#         --trace_time \
#         --encoder_layers 10 \
#         --decoder_layers 14 \
#         --embedding_size 512 \
#         --hidden_size 2048 \
#         --num_heads 8 \

# python ../src/main.py Transformer \
#         --epochs 120 \
#         --batch_size 256 \
#         --lr 0.0001 \
#         --save_path ../ckpt/Transformer85M.pt \
#         --save_model \
#         --trace_time \
#         --encoder_layers 8 \
#         --decoder_layers 12 \
#         --embedding_size 512 \
#         --hidden_size 2048 \
#         --num_heads 8 \

python ../src/main.py Transformer \
        --epochs 16 \
        --batch_size 128 \
        --lr 0.0001 \
        --save_path ../ckpt/Transformer68MES.pt \
        --save_model \
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
        --save_path ../ckpt/Transformer53MES.pt \
        --save_model \
        --trace_time \
        --encoder_layers 6 \
        --decoder_layers 6 \
        --embedding_size 512 \
        --hidden_size 2048 \
        --num_heads 8 \

