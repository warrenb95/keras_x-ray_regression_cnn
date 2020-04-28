from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime
from keras import backend as k_back
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":
    # Testing section
    # ---------------------------------------------------------------------
    # model = regression_trainer.train_new('elbow')
    # ---------------------------------------------------------------------

    # Uncomment to run the GUI Application
    # ---------------------------------------------------------------------
    # controller = Controller()
    # controller.run()
    # ---------------------------------------------------------------------

    # Uncomment to train the regression models
    # ---------------------------------------------------------------------
    # trainer = regression_trainer.Regression_Trainer().getInstance()
    # model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
    # amount_of_models = 9
    # for m in model_list:
    #     trainer.train_new(m, amount_of_models)
    #     k_back.clear_session()
    # ---------------------------------------------------------------------

    # Uncomment to train classification model
    # ---------------------------------------------------------------------
    # model = classification_trainer.train_old()
    # ---------------------------------------------------------------------

    # Uncomment to evaluiate the classification model
    # ---------------------------------------------------------------------
    trainer = classification_trainer.Classification_Trainer.getInstance()
    trainer.evaluate_model()
    # ---------------------------------------------------------------------

    # Uncomment this to test regression validation
    # ---------------------------------------------------------------------
    # regression_trainer.validate("elbow")
    # ---------------------------------------------------------------------

    # Uncomment to evaluate the regression models
    # ---------------------------------------------------------------------
    # trainer = regression_trainer.Regression_Trainer().getInstance()
    # model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
    # amount_of_models = 9
    # # model_list = ['elbow']
    # # amount_of_models = 1

    # evaluation_dict = {}

    # for m in model_list:
    #     evaluation_list = []
    #     for i in range(amount_of_models):
    #         evaluation_list.append(trainer.evaluate_model(m, i))
    #         k_back.clear_session()

    #     evaluation_dict[m] = evaluation_list

    # print(evaluation_dict)
    # with open('evaluation.txt', 'w') as eval_file:
    #     for key, values in evaluation_dict.items():
    #         eval_file.write(key)

    #         for i in range(len(values)):
    #             eval_file.write('\n\tModel - ' + str(i) + '\tLoss - ' + str(values[i]))

    # ---------------------------------------------------------------------

    # This will turn the PC off, use when training overnight
    # ---------------------------------------------------------------------
    # os.system("shutdown /s /t 1")
    # This will sleep the PC
    # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    # ---------------------------------------------------------------------
    
