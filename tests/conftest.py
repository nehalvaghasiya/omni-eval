import pytest

from omni_eval.fixtures.fixture_predictions import classification_fixture


@pytest.fixture
def classifier():
    return classification_fixture()
