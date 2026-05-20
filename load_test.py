import asyncio
import aiohttp
import time

URL = "http://localhost:5005/webhooks/rest/webhook"
TOTAL_REQUESTS = 1000

async def send_message(session, user_id):
    payload = {
        "sender": f"test_user_{user_id}",
        "message": "hello"
    }
    
    start_time = time.time()
    try:
        async with session.post(URL, json=payload) as response:
            await response.json()
            latency = time.time() - start_time
            return latency
    except Exception as e:
        return None

async def main():
    print(f" Launching {TOTAL_REQUESTS} simultaneous requests straight to the C++ Tracker Store...")
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = [send_message(session, i) for i in range(TOTAL_REQUESTS)]
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_latencies = [l for l in latencies if l is not None]
        
        print("\n--- PERFORMANCE BENCHMARK RESULTS ---")
        print(f"Total Time for {TOTAL_REQUESTS} requests: {total_time:.2f} seconds")
        print(f"Successful Responses: {len(successful_latencies)}/{TOTAL_REQUESTS}")
        if successful_latencies:
            print(f"Average Latency per request: {sum(successful_latencies)/len(successful_latencies):.4f} seconds")
            print(f"Max Latency: {max(successful_latencies):.4f} seconds")

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())