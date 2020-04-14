from controller import Controller
import classification_trainer
import regression_trainer
from datetime import datetime
from keras import backend as k_back
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

    # Uncomment this to test regression validation
    # ---------------------------------------------------------------------
    # regression_trainer.validate("elbow")
    # ---------------------------------------------------------------------

    # Uncomment to evaluate the regression models
    # ---------------------------------------------------------------------
    trainer = regression_trainer.Regression_Trainer().getInstance()
    model_list = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']
    amount_of_models = 3
    # model_list = ['elbow']
    # amount_of_models = 1

    evaluation_dict = {}

    for m in model_list:
        for i in range(amount_of_models):
            loss_eval = trainer.evaluate_model(m, i)

            while loss_eval >= 0.25:
                trainer.train_new(m, i)
                k_back.clear_session()
                loss_eval = trainer.evaluate_model(m, i)

            print(f'{str(m).capitalize} - {str(i)} Training loss <= 0.25')

    print(evaluation_dict)
    
    # ---------------------------------------------------------------------

    # This will turn the PC off, use when training overnight
    # ---------------------------------------------------------------------
    # os.system("shutdown /s /t 1")
    # This will sleep the PC
    # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    # ---------------------------------------------------------------------
    
