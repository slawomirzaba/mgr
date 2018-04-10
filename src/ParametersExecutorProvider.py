import sys
import getopt
import constants


class ParametersExecutorProvider:
    def __init__(self):
        self.help = constants.HELP_INSTRUCTIONS

        # default values
        self.algorithm_name = 'LevenbergMarquardt'
        self.test_size = 0.7
        self.learning_rate = 0.01
        self.epochs_number = 30
        self.hidden_units_number = 3
        self.hidden_layers_number = 1

    def __raise_error(self):
        print(self.help)
        sys.exit(2)

    def __translate_opts(self, opts):
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(self.help)
                sys.exit()
            elif opt in ("-a", "--algorithm_name"):
                self.algorithm_name = arg
                if self.algorithm_name not in constants.ALGORITHMS.keys():
                    print('\nUNSUPPORTED  ALGORITHM!\n')
                    self.__raise_error()

            elif opt in ("-s", "--test_size"):
                self.test_size = float(arg)
                if self.test_size < 0.1 or self.test_size > 0.9:
                    print('\nOUT OF RANGE TEST SIZE!\n')
                    self.__raise_error()

            elif opt in ("-l", "--learning_rate"):
                self.learning_rate = float(arg)
                if self.learning_rate < 0.01 or self.learning_rate > 0.99:
                    print('\nOUT OF RANGE LEARNING RATE!\n')
                    self.__raise_error()

            elif opt in ("-e", "--epochs_number"):
                self.epochs_number = int(arg)
                if self.epochs_number < 1:
                    print('\nOUT OF RANGE EPOCHS!\n')
                    self.__raise_error()

            elif opt in ("-u", "--hidden_units_number"):
                self.hidden_units_number = int(arg)
                if self.hidden_units_number < 1:
                    print('\nOUT OF RANGE HIDDEN UNITS NUMBER!\n')
                    self.__raise_error()
            
            elif opt in ("-i", "--hidden_layers_number"):
                self.hidden_layers_number = int(arg)
                if self.hidden_layers_number < 1:
                    print('\nOUT OF RANGE HIDDEN LAYERS NUMBER!\n')
                    self.__raise_error()

    def get_parameters(self):
        argv = sys.argv[1:]
        try:
            opts, _ = getopt.getopt(argv, "ha:s:l:e:u:i:", ["help", "algorithm_name=", "test_size=",
                                                          "learning_rate=", "epochs_number=", 
                                                          "hidden_units_number=", "hidden_layers_number="])
        except getopt.GetoptError:
            self.__raise_error()

        self.__translate_opts(opts)
        return {
            'algorithm_name': self.algorithm_name,
            'test_size': self.test_size,
            'learning_rate': self.learning_rate,
            'epochs_number': self.epochs_number,
            'hidden_units_number': self.hidden_units_number,
            'hidden_layers_number': self.hidden_layers_number
        }
