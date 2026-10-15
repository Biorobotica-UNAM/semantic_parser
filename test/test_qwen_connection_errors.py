"""Transport failures have a machine-readable code distinct from command errors."""
import json
from unittest.mock import Mock

import httpx
import pytest

from semantic_parser.qwen_semantic_node import QwenSemanticNode


@pytest.mark.parametrize('error, code', [
    (httpx.ConnectError('[Errno 111] Connection refused'), 'gpsr_unavailable'),
    (httpx.ReadTimeout('timed out'), 'gpsr_unavailable'),
    (ConnectionRefusedError('Connection refused'), 'gpsr_unavailable'),
    (RuntimeError('unsupported command'), 'semantic_error'),
])
def test_translate_error_code(error, code):
    node = Mock()
    node._client.translate.side_effect = error
    result = json.loads(QwenSemanticNode._translate_command(node, 'Go to the exit.'))
    assert result['status'] == 'aborted'
    assert result['error_code'] == code
    assert result['reason'] == str(error)

