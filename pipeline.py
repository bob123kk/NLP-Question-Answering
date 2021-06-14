from preprocessor import Pipeline
import config

pipeline = Pipeline(path=config.path,
                    size=config.size,
                    test_size=config.test_size,
                    methods=config.methods,
                    w2v_model_path=config.w2v_model_path,
                    lgb_params=config.lgb_params
                    )

if __name__ == '__main__':
    # load and split data
    pipeline.train_test_split_here()

    train_data = pipeline.train_data
    test_data = pipeline.test_data
    print('Finish reading data')

    processed_train = pipeline.preprocess(train_data)
    processed_test = pipeline.preprocess(test_data)
    print('Finish processing data')

    pipeline.fit(processed_train)
    print('model performance')
    pipeline.evaluate_model(processed_train)

    predictions = pipeline.predict(processed_test)
    print(predictions)
