import pytest


def pytest_configure(config):
    """ Permanently exclude folders from being searched for tests """
    config.addinivalue_line('norecursedirs', '__pycache__')
    config.addinivalue_line('norecursedirs', 'data')


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="Run tests requiring online access or live trading")
    parser.addoption("--livedb", action="store_true", default=False, help="Run tests requiring the full database")
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption("--seed", action="store", type=int, default=0, help="Seed for randomly generated mockups")


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None and not call.excinfo.errisinstance((pytest.xfail.Exception, pytest.skip.Exception)):
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    for mark in ('live', 'livedb', 'slow'):
        if mark in item.keywords and not item.config.getoption("--" + mark):
            pytest.skip("need --%s option to run" % mark)

    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)
