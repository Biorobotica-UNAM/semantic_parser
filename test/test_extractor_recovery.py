"""Exercise spoken recovery answers independently of ROS and ASR."""

from pathlib import Path

import pytest
import yaml

from semantic_parser.extractor import SentenceEntityExtractor


@pytest.fixture
def extractor(tmp_path):
    dictionary = Path(__file__).resolve().parents[3] / 'Knowledge/knowledge_bank/dictionaries/confirm.yaml'
    (tmp_path / 'confirm.yaml').write_text(dictionary.read_text())
    (tmp_path / 'order.yaml').write_text(yaml.safe_dump({
        'label': 'ORDER', 'patterns': ['milk', 'chocolate milk'],
    }))
    return SentenceEntityExtractor(tmp_path)


@pytest.mark.parametrize('speech, expected', [
    ('Kitchen bar.', ['kitchen_bar']),
    ('Please go to the kitchen bar.', ['kitchen_bar']),
    ('Could you go to the cooking table?', ['cooking_table']),
    ('I would like you to navigate to the dining room.', ['dining_room']),
    ('The destination is the bedroom', ['bedroom']),
    ('I meant kitchen_bar', ['kitchen_bar']),
    ('“New custom place.”', ['new_custom_place']),
    ('Go to the kitchen or the bedroom', None),
    ('not the kitchen', None),
    ('', None),
])
def test_location(extractor, speech, expected):
    assert extractor.extract(speech, 'LOCATION').value == expected


@pytest.mark.parametrize('speech, expected', [
    ('Yes, that is correct.', 'yes'),
    ('No, that is wrong.', 'no'),
    ("That's not correct", 'no'),
    ("That isn't right", 'no'),
    ('not confirmed', 'no'),
    ('Yes, no.', None),
    ('No, that is correct.', None),
    ('I am not sure', None),
    ('', None),
])
def test_confirmation(extractor, speech, expected):
    assert extractor.extract(speech, 'CONFIRM').value == expected


def test_other_categories_preserve_longest_match(extractor):
    assert extractor.extract('Bring chocolate milk.', 'ORDER').value == ['chocolate_milk']
