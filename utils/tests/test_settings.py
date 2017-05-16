import pytest
from .. import settings

@pytest.fixture(scope='module')
def config():
    cfg = settings.get_config()
    if not cfg:
        pytest.skip("No configuration file present")
    return cfg


def test_bfaccounts(config):
    accounts = config.accounts.betfair
    for acc in accounts:
        assert acc.username
        assert acc.password
        assert acc.strategy
        assert not acc.disabled


def test_irc(config):
    irc = config.irc
    assert irc.server
    assert irc.name
    assert irc.auth


def test_general(config):
    general = config.general
    assert general.basedir
    assert general.max_mem
    assert general.admin_mail


def test_strategies(config):
    vb = config.strategies.valuebetting
    assert 'closing_only' in vb
    assert vb.fractional_kelly_back
    assert vb.fractional_kelly_lay


def test_configwalk():
    class C(object):
        def __init__(self, **kwargs):
            self._dict = dict()
            self._dict.update(kwargs)

    b = C(x=C(bla=4))
    conv = settings._config_walk(b, object_type=C, dict_convert=lambda x:x._dict)
    assert isinstance(conv, dict)
    assert isinstance(conv['x'], dict)
    assert conv['x']['bla'] == 4
