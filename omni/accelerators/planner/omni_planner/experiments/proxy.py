from fastapi import FastAPI, Request, Response  # Added Response here
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

# Upstream server URL (pointing to the mock server)
UPSTREAM_URL = "http://127.0.0.1:8080"

async def stream_chunked_response(client: httpx.AsyncClient, url: str):
    """Stream the upstream response, respecting chunk boundaries."""
    async with client.stream("GET", url) as response:
        # Yield raw bytes to preserve chunk boundaries
        async for chunk in response.aiter_raw():
            yield chunk

@app.get("/{path:path}", response_class=StreamingResponse)
async def reverse_proxy(request: Request, path: str):
    """Reverse proxy endpoint that preserves chunk boundaries."""
    # Construct the upstream URL
    upstream_url = f"{UPSTREAM_URL}/{path}"
    print(f"Proxying to {upstream_url}")
    if request.query_params:
        upstream_url += f"?{request.query_params}"

    # Async HTTP client
    async with httpx.AsyncClient() as client:
        try:
            # Use stream context to get the response
            async with client.stream("GET", upstream_url) as response:
                headers = response.headers
                status_code = response.status_code

                # Stream the response directly
                return StreamingResponse(
                    stream_chunked_response(client, upstream_url),
                    status_code=status_code,
                    headers=headers,
                    media_type=headers.get("Content-Type", "application/octet-stream")
                )
        except httpx.RequestError as e:
            return Response(content=f"Error: {str(e)}", status_code=502)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)