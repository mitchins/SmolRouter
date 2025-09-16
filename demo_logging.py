#!/usr/bin/env python3
"""
Demo script to test the logging functionality of the OpenAI Model Rerouter
"""

import asyncio
import httpx

async def demo_requests():
    """Send some demo requests to test logging"""
    
    base_url = "http://127.0.0.1:1234"
    
    print("🚀 Testing OpenAI Model Rerouter with Request Logging")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # Test 1: OpenAI Chat Completion (non-streaming)
        print("1️⃣  Testing OpenAI Chat Completion...")
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": False
                },
                timeout=10
            )
            print(f"   Status: {response.status_code}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test 2: Ollama Generate
        print("\n2️⃣  Testing Ollama Generate...")
        try:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": "Tell me a joke",
                    "stream": False
                },
                timeout=10
            )
            print(f"   Status: {response.status_code}")
        except Exception as e:
            print(f"   Error: {e}")
            
        # Test 3: Model listing
        print("\n3️⃣  Testing Model Listing...")
        try:
            response = await client.get(f"{base_url}/v1/models", timeout=10)
            print(f"   Status: {response.status_code}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Wait a moment for logs to be written
        await asyncio.sleep(1)
        
        # Check the dashboard
        print("\n📊 Checking Dashboard...")
        try:
            response = await client.get(f"{base_url}/api/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"   Total Requests: {stats['total_requests']}")
                print(f"   OpenAI Requests: {stats['openai_requests']}")
                print(f"   Ollama Requests: {stats['ollama_requests']}")
                print(f"   Recent (24h): {stats['recent_requests']}")
            else:
                print(f"   Dashboard Error: {response.status_code}")
        except Exception as e:
            print(f"   Dashboard Error: {e}")
        
        # Get recent logs
        print("\n📝 Recent Request Logs:")
        try:
            response = await client.get(f"{base_url}/api/logs?limit=5")
            if response.status_code == 200:
                logs = response.json()
                print(f"   Found {len(logs)} recent requests:")
                for log in logs:
                    print(f"   • {log['timestamp'][:19]} | {log['service_type'].upper()} | "
                          f"{log['path']} | {log['status_code']} | {log['duration_ms']}ms")
            else:
                print(f"   Logs Error: {response.status_code}")
        except Exception as e:
            print(f"   Logs Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! Visit http://127.0.0.1:1234 to see the web UI")

if __name__ == "__main__":
    asyncio.run(demo_requests())