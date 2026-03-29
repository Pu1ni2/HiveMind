"""Start the YCONIC server."""

import uvicorn

if __name__ == "__main__":
    print("\n  YCONIC Server starting at http://localhost:8000\n")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
