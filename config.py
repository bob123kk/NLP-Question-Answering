path ='simplified-nq-train.jsonl'
w2v_model_path = 'GoogleNews.bin'
size = 10
test_size = 0.33
methods = ['euclidean','cosine','chebyshev','correlation']
lgb_params={
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_iterations ': 200,
            'max_depth': 7,
            'num_leaves': 80,
            'learning_rate': 0.01, 
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'xgboost_dart_mode': True,
            'verbose': -1, 
            'is_unbalance': False,
            'num_threads': 5
            }