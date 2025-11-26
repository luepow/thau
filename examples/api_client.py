#!/usr/bin/env python3
"""
API Client Example
Shows how to use THAU through REST API
"""

import requests
import json
from typing import Optional

API_BASE_URL = "http://localhost:8000"


def chat_message(message: str, conversation_id: Optional[str] = None) -> dict:
    """Send a chat message to THAU API"""
    url = f"{API_BASE_URL}/chat/message"

    payload = {
        "message": message,
        "max_new_tokens": 100,
        "temperature": 0.7
    }

    if conversation_id:
        payload["conversation_id"] = conversation_id

    response = requests.post(url, json=payload)
    return response.json()


def get_chat_history(conversation_id: str) -> dict:
    """Get conversation history"""
    url = f"{API_BASE_URL}/chat/history/{conversation_id}"
    response = requests.get(url)
    return response.json()


def train_interaction(text: str, steps: int = 10) -> dict:
    """Train THAU with new interaction"""
    url = f"{API_BASE_URL}/train/interaction"

    payload = {
        "text": text,
        "steps": steps,
        "learning_rate": 3e-4
    }

    response = requests.post(url, json=payload)
    return response.json()


def get_training_stats() -> dict:
    """Get training statistics"""
    url = f"{API_BASE_URL}/train/stats"
    response = requests.get(url)
    return response.json()


def store_memory(content: str, importance: int = 5) -> dict:
    """Store something in THAU's memory"""
    url = f"{API_BASE_URL}/memory/store"

    payload = {
        "content": content,
        "importance": importance
    }

    response = requests.post(url, json=payload)
    return response.json()


def recall_memory(query: str, n_results: int = 3) -> dict:
    """Recall memories related to query"""
    url = f"{API_BASE_URL}/memory/recall"

    params = {
        "query": query,
        "n_results": n_results
    }

    response = requests.get(url, params=params)
    return response.json()


def main():
    print("="*60)
    print("🌐 THAU API Client Example")
    print("="*60)
    print()

    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print("✅ API is running")
        print(f"   Status: {response.json()}")
        print()
    except requests.exceptions.ConnectionError:
        print("❌ API is not running!")
        print("   Start it with: python api/thau_code_server.py")
        return

    # Example 1: Simple chat
    print("📝 Example 1: Simple Chat")
    print("-" * 40)
    result = chat_message("Hello, THAU! Tell me about yourself.")
    print(f"🤖 THAU: {result['response']}")
    conversation_id = result['conversation_id']
    print(f"   Conversation ID: {conversation_id}")
    print()

    # Example 2: Continue conversation
    print("📝 Example 2: Continue Conversation")
    print("-" * 40)
    result = chat_message(
        "What can you do?",
        conversation_id=conversation_id
    )
    print(f"🤖 THAU: {result['response']}")
    print()

    # Example 3: Get chat history
    print("📝 Example 3: Get Chat History")
    print("-" * 40)
    history = get_chat_history(conversation_id)
    print(f"Messages: {len(history['history'])}")
    for msg in history['history']:
        role = msg['role'].upper()
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"   {role}: {content}")
    print()

    # Example 4: Store memory
    print("📝 Example 4: Store Memory")
    print("-" * 40)
    memory_result = store_memory(
        "THAU is an AI system with cognitive growth capabilities",
        importance=8
    )
    print(f"✅ Memory stored: {memory_result['memory_id']}")
    print()

    # Example 5: Recall memory
    print("📝 Example 5: Recall Memory")
    print("-" * 40)
    recall_result = recall_memory("cognitive growth")
    print(f"Found {len(recall_result['results'])} memories:")
    for mem in recall_result['results']:
        print(f"   - {mem['content'][:60]}...")
        print(f"     (similarity: {mem['similarity']:.3f})")
    print()

    # Example 6: Training stats
    print("📝 Example 6: Training Statistics")
    print("-" * 40)
    stats = get_training_stats()
    print(f"📊 Training Stats:")
    print(f"   Total interactions: {stats.get('total_interactions', 0)}")
    print(f"   Total trainings: {stats.get('total_trainings', 0)}")
    print(f"   Current age: {stats.get('current_age', 0)}")
    print()

    # Example 7: Train with interaction (optional, commented by default)
    # print("📝 Example 7: Train with New Interaction")
    # print("-" * 40)
    # train_result = train_interaction(
    #     "Machine learning is a fascinating field of AI",
    #     steps=5
    # )
    # print(f"✅ Training completed")
    # print(f"   Final loss: {train_result['final_loss']:.4f}")
    # print()

    print("="*60)
    print("✅ All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
