class Model():
    def __init__(self):
        self.__base_model = None
        self.__model_microanurasim = None
        self.__model_himo_and_hardexudates = None
        self.__model_combine = None
        self.__model_numarical_features = None
        pass

    def base_model(self):
        '''
        this function combine the
        :return:
        '''
        pass

    def load_model_from_json(self):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

