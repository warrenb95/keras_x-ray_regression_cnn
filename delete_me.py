import regression_trainer
import classification_trainer
import helper_funcs

class_trainer = classification_trainer.Classification_Trainer().getInstance()
regres_trainer = regression_trainer.Regression_Trainer().getInstance()

model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
amount_of_models = 9

for m in model_list:
    normal_count = 0
    abnormal_count = 0
    normal_paths = []
    abnormal_paths = []

    valid_dataset_file = 'dataset/' + 'train_' + m + '.csv'
    df = helper_funcs.load_dataset_attributes(valid_dataset_file)

    for i in range(df['path'].size):
        if normal_count == 5 and abnormal_count == 5:
            break

        if (normal_count == 5 and df['target'].get(i) == 0) or (abnormal_count == 5 and df['target'].get(i) == 1.0):
            continue
        
        class_result, cur_image = class_trainer.predict_classification(df['path'].get(i))

        prediction = -100

        if class_result == 0:
            prediction = regres_trainer.predict(cur_image, 'elbow', amount_of_models)
        elif class_result == 1:
            prediction = regres_trainer.predict(cur_image, 'finger', amount_of_models)
        elif class_result == 2:
            prediction = regres_trainer.predict(cur_image, 'forearm', amount_of_models)
        elif class_result == 3:
            prediction = regres_trainer.predict(cur_image, 'hand', amount_of_models)
        elif class_result == 4:
            prediction = regres_trainer.predict(cur_image, 'humerus', amount_of_models)
        elif class_result == 5:
            prediction = regres_trainer.predict(cur_image, 'shoulder', amount_of_models)
        elif class_result == 6:
            prediction = regres_trainer.predict(cur_image, 'wrist', amount_of_models)
        else:
            print(f'Invalid class_result {class_result}')

        if df['target'].get(i) == 0 and normal_count <= 5 and prediction < 15.0:
            normal_paths.append(df['path'].get(i))
            normal_count += 1
        elif df['target'].get(i) == 1.0 and abnormal_count <= 5 and prediction > 75.0:
            abnormal_paths.append(df['path'].get(i))
            abnormal_count += 1


    with open('valid.txt', 'w') as valid_file:
        for path in normal_paths:
            valid_file.write(path)
        
        for path in abnormal_paths:
            valid_file.write(path)