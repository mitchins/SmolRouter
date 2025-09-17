#!/usr/bin/env python3
"""
Test JSON markdown fencing variations
Relocated into tests/.
"""

import sys
sys.path.insert(0, '.')

from smolrouter.app import strip_json_markdown_from_text


def test_json_fencing_variations():
    """Test both backtick and square bracket JSON fencing"""
    
    # Test cases for different fencing patterns
    test_cases = [
        # Original backtick pattern
        {
            "name": "Backtick fencing (original)",
            "input": 'Here is some JSON: ```json\n{\n  "commit_message": "test"\n}\n```',
            "expected": 'Here is some JSON: { "commit_message": "test" }'
        },
        
        # New square bracket pattern (discovered variation)
        {
            "name": "Square bracket fencing (new)",
            "input": '[json] { "commit_message": "feat: add advanced routing, JWT authentication, blob storage, and enhanced security features" } [json]',
            "expected": '{ "commit_message": "feat: add advanced routing, JWT authentication, blob storage, and enhanced security features" }'
        },
    ]
    
    for test_case in test_cases:
        result = strip_json_markdown_from_text(test_case['input'])
        assert result == test_case['expected']


def test_real_world_example():
    real_example = '[json] { "commit_message": "feat: add advanced routing, JWT authentication, blob storage, and enhanced security features" } [json]'
    expected = '{ "commit_message": "feat: add advanced routing, JWT authentication, blob storage, and enhanced security features" }'
    result = strip_json_markdown_from_text(real_example)
    assert result == expected
