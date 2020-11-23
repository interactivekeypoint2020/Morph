from torch import optim

def get_optimizer(config, model):
    optimizer = base_optimizer(config, model)
    return optimizer

class base_optimizer(object):
    def __init__(self, config, model):
        super(base_optimizer, self).__init__()
        if config.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        else:
            raise

        if config.scheduler == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, config.criterion.state, verbose=True)
        else:
            self.scheduler = None

    def zero_grad(self):
        self.optimizer.zero_grad()
        return None

    def update_model(self, loss, mode_idx):
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None