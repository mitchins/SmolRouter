#!/usr/bin/env python3
"""
Simple test to demonstrate the new SOLID architecture in action.

This test shows how the new architecture works with model aggregation,
aliasing, and clean separation of concerns.
"""

import asyncio
import logging
from smolrouter.interfaces import ProviderConfig, ClientContext
from smolrouter.providers import ProviderFactory
from smolrouter.caching import InMemoryModelCache
from smolrouter.strategies import SmartModelStrategy
from smolrouter.access_control import NoAccessControl
from smolrouter.mediator import ModelMediator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_new_architecture():
    """Demonstrate the new architecture with mock providers"""
    
    print("🚀 SmolRouter New Architecture Demo")
    print("=" * 50)
    
    # Step 1: Create some mock providers
    print("\n📡 Creating mock providers...")
    
    # These would normally be real endpoints, but for demo we'll use mock URLs
    provider_configs = [
        ProviderConfig(
            name="fast-kitten",
            type="openai", 
            url="http://localhost:8001",  # Mock fast server
            priority=0,
            enabled=True
        ),
        ProviderConfig(
            name="slow-kitten", 
            type="openai",
            url="http://localhost:8002",  # Mock slow server  
            priority=1,
            enabled=True
        ),
        ProviderConfig(
            name="gpu-server",
            type="ollama",
            url="http://localhost:11434", # Mock Ollama server
            priority=2, 
            enabled=True
        )
    ]
    
    providers = []
    for config in provider_configs:
        try:
            provider = ProviderFactory.create_provider(config)
            providers.append(provider)
            print(f"  ✓ Created {config.name} ({config.type}) -> {config.url}")
        except Exception as e:
            print(f"  ✗ Failed to create {config.name}: {e}")
    
    # Step 2: Create architecture components
    print("\n🏗️  Building architecture components...")
    
    # Cache for performance
    cache = InMemoryModelCache(default_ttl=300)
    print("  ✓ Created in-memory cache (TTL: 300s)")
    
    # Strategy for model resolution and aliasing
    strategy_config = {
        'model_map': {
            'gpt-4': 'llama3-70b',
            'gpt-3.5-turbo': 'llama3-8b'
        },
        'servers': {
            'fast-kitten': 'http://localhost:8001',
            'slow-kitten': 'http://localhost:8002',
            'gpu-server': 'http://localhost:11434'
        },
        'provider_priorities': {
            'fast-kitten': 0,
            'slow-kitten': 1, 
            'gpu-server': 2
        }
    }
    strategy = SmartModelStrategy(strategy_config)
    print("  ✓ Created smart model strategy with aliases")
    
    # Access control (no restrictions for demo)
    access_control = NoAccessControl()
    print("  ✓ Created no-op access control")
    
    # Aggregator for model discovery
    from smolrouter.caching import ModelAggregator
    aggregator = ModelAggregator(providers, cache, default_cache_ttl=300)
    print("  ✓ Created model aggregator")
    
    # Central mediator
    mediator = ModelMediator(aggregator, strategy, access_control)
    print("  ✓ Created model mediator")
    
    # Step 3: Test architecture in action
    print("\n🎯 Testing architecture features...")
    
    # Create a mock client
    client = ClientContext(ip="192.168.1.50", auth_payload=None)
    print(f"  📱 Mock client: {client.ip}")
    
    try:
        # Test 1: Get available models (this will try to connect to mock servers)
        print("\n  🔍 Discovering available models...")
        try:
            models = await mediator.get_available_models(client, force_refresh=True)
            print(f"     Found {len(models)} models")
            
            for model in models[:3]:  # Show first 3
                print(f"     - {model.display_name} ({model.provider_type})")
        except Exception as e:
            print(f"     ⚠️  Model discovery failed (expected with mock servers): {e}")
            
        # Test 2: Test model resolution
        print("\n  🎯 Testing model resolution...")
        test_requests = [
            "gpt-4",  # Should resolve via alias
            "llama3-70b [fast-kitten]",  # Fully qualified name
            "coding-model"  # Non-existent model
        ]
        
        for requested_model in test_requests:
            try:
                resolved = await mediator.resolve_model_for_request(requested_model, client)
                if resolved:
                    print(f"     ✓ '{requested_model}' -> '{resolved.display_name}'")
                else:
                    print(f"     ✗ '{requested_model}' -> Not found")
            except Exception as e:
                print(f"     ⚠️  '{requested_model}' -> Error: {e}")
        
        # Test 3: Get provider health
        print("\n  💓 Checking provider health...")
        health = await mediator.get_provider_health()
        for provider_id, is_healthy in health.items():
            status = "🟢 Healthy" if is_healthy else "🔴 Unhealthy"
            print(f"     {provider_id}: {status}")
        
        # Test 4: Get architecture stats
        print("\n  📊 Architecture statistics...")
        stats = await mediator.get_mediator_stats()
        print(f"     Providers: {stats['aggregation']['provider_count']}")
        print(f"     Cache enabled: {stats['aggregation']['cache_stats'].get('total_entries', 'N/A')} entries")
        print(f"     Strategy: {stats['strategy_type']}")
        print(f"     Access control: {stats['access_control_type']}")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        mediator.close()
        print("  ✓ Architecture shut down cleanly")
    
    print("\n✅ Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  • Model aggregation from multiple providers")
    print("  • Smart caching with TTL")
    print("  • Model aliasing and resolution")
    print("  • Provider health monitoring") 
    print("  • Clean separation of concerns (SOLID principles)")
    print("  • Graceful error handling and fallbacks")


if __name__ == "__main__":
    asyncio.run(demo_new_architecture())