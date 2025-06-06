from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def fake_chunked_stream():
    """Yields a chunked response with delays to simulate streaming."""
    # Chunk 1: "Hello"
    yield b"5\r\nHello\r\n"
    await asyncio.sleep(1)  # 1-second delay
    # Chunk 2: "World"
    yield b"5\r\nWorld\r\n"
    await asyncio.sleep(1)  # 1-second delay
    # End of stream
    yield b"0\r\n\r\n"

@app.get("/stream")
async def stream_endpoint():
    """Endpoint that returns a chunked response."""
    return StreamingResponse(
        fake_chunked_stream(),
        headers={"Transfer-Encoding": "chunked"},
        media_type="text/plain"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)