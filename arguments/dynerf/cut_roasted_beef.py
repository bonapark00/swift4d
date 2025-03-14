_base_ = './default.py'
OptimizationParams = dict( # 10000 轮达到最大·可以尝试设置1w
    hash_init_lr = 0.0002 ,
    hash_final_lr = 0.00002 ,
    hashmap_size = 19 ,
    # opacity_reset_interval = 3000 ,
)
