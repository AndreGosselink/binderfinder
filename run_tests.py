import pytest

ret = pytest.main(args=['tests'])

if ret != 0:
    raise Exception('pytest failed!')
