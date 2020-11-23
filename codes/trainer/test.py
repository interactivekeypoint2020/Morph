from trainer.run_hint_test import run_hint_test
from misc import util


def test(config, model, test_loader, criterion, writer=None, split='test'):
    util.log(config, '================== {} started ==================='.format(split))

    test_results = run_hint_test(config=config, model=model, grad=False, criterion=criterion, loader=test_loader, split=split)

    for key in test_results:
        util.log(config, '{} {} | {:.6f}'.format(split.upper(), key, test_results[key]))

    util.log(config, '================== {} finished ==================='.format(split))
