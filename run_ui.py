"""Launcher for the ChaosRL interactive web visualizer and benchmark portal."""

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  ChaosRL: Autonomous Microservice Chaos Engineering Portal")
    print("  Serving at: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run("chaosrl.ui.server:app", host="127.0.0.1", port=8000, reload=False)
