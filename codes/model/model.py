from model.ours import Ours

def get_model(config):
    # ===================
    # model init
    # ===================

    model = Ours(config)
    return model