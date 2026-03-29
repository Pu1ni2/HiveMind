"""Start the YCONIC server."""

import uvicorn

if __name__ == "__main__":
    print()
    print("  ╔════════════════════════════════════════╗")
    print("  ║   YCONIC — Agent Orchestration Engine  ║")
    print("  ║   http://localhost:8000                 ║")
    print("  ╚════════════════════════════════════════╝")
    print()
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
