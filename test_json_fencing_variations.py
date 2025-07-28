#!/usr/bin/env python3
"""
Test JSON markdown fencing variations
"""

import sys
sys.path.insert(0, '.')

from smolrouter.app import strip_json_markdown_from_text

def test_json_fencing_variations():
    """Test both backtick and square bracket JSON fencing"""
    
    print("🔧 Testing JSON Fencing Variations...")
    
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
        
        # Mixed content with both patterns
        {
            "name": "Mixed patterns",
            "input": 'Backtick: ```json\n{"type": "backtick"}\n``` and square: [json] {"type": "square"} [json]',
            "expected": 'Backtick: {"type": "backtick"} and square: {"type": "square"}'
        },
        
        # Multiple square bracket patterns
        {
            "name": "Multiple square brackets",
            "input": '[json] {"first": true} [json] and then [json] {"second": false} [json]',
            "expected": '{"first": true} and then {"second": false}'
        },
        
        # Square brackets with whitespace
        {
            "name": "Square brackets with whitespace",
            "input": '[json]   {  "spaced": "content"  }   [json]',
            "expected": '{ "spaced": "content" }'
        },
        
        # No JSON fencing (should remain unchanged)
        {
            "name": "No fencing",
            "input": 'Just regular text with {"json": "but no fencing"}',
            "expected": 'Just regular text with {"json": "but no fencing"}'
        },
        
        # Edge case: malformed square brackets
        {
            "name": "Malformed square brackets",
            "input": '[json {"missing": "closing"} json]',
            "expected": '[json {"missing": "closing"} json]'  # Should remain unchanged
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        try:
            result = strip_json_markdown_from_text(test_case['input'])
            
            if result == test_case['expected']:
                print("✅ PASSED")
                print(f"   Input:    {repr(test_case['input'])}")
                print(f"   Output:   {repr(result)}")
                passed += 1
            else:
                print("❌ FAILED")
                print(f"   Input:    {repr(test_case['input'])}")
                print(f"   Expected: {repr(test_case['expected'])}")
                print(f"   Got:      {repr(result)}")
                failed += 1
                
        except Exception as e:
            print(f"💥 CRASHED: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All JSON fencing variations work correctly!")
        print("📝 The discovered [json] ... [json] pattern is now supported")
    else:
        print("⚠️  Some tests failed - JSON stripping may not work correctly")
    
    return failed == 0

def test_real_world_example():
    """Test the specific example that was discovered"""
    print("\n🌍 Testing Real-World Example...")
    
    # The actual example from the log
    real_example = '[json] { "commit_message": "feat: add advanced routing, JWT authentication, blob storage, and enhanced security features" } [json]'
    expected = '{ "commit_message": "feat: add advanced routing, JWT authentication, blob storage, and enhanced security features" }'
    
    result = strip_json_markdown_from_text(real_example)
    
    print(f"Input:  {repr(real_example)}")
    print(f"Output: {repr(result)}")
    
    if result == expected:
        print("✅ Real-world example works correctly!")
        return True
    else:
        print("❌ Real-world example failed!")
        print(f"Expected: {repr(expected)}")
        return False

def main():
    """Run all JSON fencing tests"""
    print("🧪 Testing JSON Markdown Fencing Variations\n")
    
    basic_tests_pass = test_json_fencing_variations()
    real_world_pass = test_real_world_example()
    
    overall_success = basic_tests_pass and real_world_pass
    
    print(f"\n{'='*60}")
    print("🏁 OVERALL RESULTS")
    print(f"{'='*60}")
    
    if overall_success:
        print("🎉 All JSON fencing tests passed!")
        print("✅ Both ```json``` and [json] patterns are supported")
        print("✅ The discovered variation is now handled correctly")
    else:
        print("❌ Some JSON fencing tests failed")
        print("🚨 JSON content may not be extracted correctly")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)