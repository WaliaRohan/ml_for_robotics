import torch.nn as nn


def light_arch():

    model = nn.Sequential()

    # Encoder layers
    model.add_module("encoder_input", nn.Linear(3, 15))
    model.add_module("en_act_1", nn.ReLU())
    model.add_module("encoder_hidden", nn.Linear(15, 10))
    model.add_module("en_act_2", nn.ReLU())
    model.add_module("encoder_output", nn.Linear(10, 5))
    # output of en_act_3 = batch_predicted_u

    # Decoder layers
    model.add_module("decoder_input", nn.Linear(5, 25))
    model.add_module("dec_act_1", nn.ReLU())
    model.add_module("decoder_hidden", nn.Linear(25, 15))
    model.add_module("dec_act_2", nn.ReLU())
    model.add_module("decoder_output", nn.Linear(15, 3))

    return model


def dense_arch():

    model = nn.Sequential()

    # Encoder layers
    model.add_module("encoder_input", nn.Linear(3, 25))
    model.add_module("en_act_1", nn.ReLU())
    model.add_module("encoder_hidden", nn.Linear(25, 15))
    model.add_module("en_act_2", nn.ReLU())
    model.add_module("encoder_output", nn.Linear(15, 5))
    # output of en_act_3 = batch_predicted_u

    # Decoder layers
    model.add_module("decoder_input", nn.Linear(5, 50))
    model.add_module("dec_act_1", nn.ReLU())
    model.add_module("decoder_hidden", nn.Linear(50, 30))
    model.add_module("dec_act_2", nn.ReLU())
    model.add_module("decoder_output", nn.Linear(30, 3))

    return model