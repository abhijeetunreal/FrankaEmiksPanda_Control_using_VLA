# VLA Franka Emika Panda

This repository contains a MuJoCo-based Vision-Language-Action (VLA) prototype for a Franka Emika Panda robot arm.

## What this project is

- A small simulation environment built on MuJoCo.
- Uses a Franka Panda robot description from `franka_emika_panda/`.
- Implements a simple VLA controller in `franka_emika_panda/vla_controller.py`.
- Accepts a natural language command, generates an action plan with an LLM, and executes it in simulation.

## Key features

- Scene rendering and camera capture using MuJoCo and OpenCV.
- Simple inverse kinematics-based motion for the Panda arm.
- Gripper open/close control.
- Language-driven commands like `move to red box`, `hover over blue box`, `close gripper`, and `move home`.
- Uses an LLM API configured in the controller to translate text + image into JSON action plans.
- **Web workspace**: a browser UI (plain HTML/CSS/JS) with a live MuJoCo viewport, chat, terminal log, and drag-to-move objects (`server/`, `web/`).

## Project structure

- `franka_emika_panda/`
  - `vla_controller.py` — main VLA controller script (CLI; also used by the web backend).
  - `sim_service.py` — headless MuJoCo loop, WebSocket command queue, object dragging.
  - `vla_scene.xml` — MuJoCo scene with the Panda robot and objects.
  - `panda.xml`, `scene.xml`, `mjx_*.xml` — robot model and scene files.
  - `README.md` — model-specific documentation for the Panda MJCF/MJX assets.
- `server/` — FastAPI app and WebSocket (`server/app.py`).
- `web/` — static workspace UI: `index.html`, `app.js`, `app.css` (no Node.js build step).
- `requirements.txt` — Python dependencies.
- `run.bat` — **Windows launcher:** creates/updates `venv`, installs dependencies, starts the API server in a new window, opens **http://127.0.0.1:8000** in your browser.
- `launcher_api.bat` — helper used by `run.bat` to run uvicorn (you can ignore it unless debugging).
- `.gitignore` — excludes temporary files, virtual environment folders, MuJoCo keys, and logs.

## Requirements

- Python 3.10+.
- MuJoCo 2.3.3 or later.
- A valid MuJoCo license key installed locally (e.g. `mjkey.txt`), if your MuJoCo build still requires it.
- GPU or CPU support for MuJoCo depending on your local setup.
- **Web UI:** no Node.js required — the interface is static files served by FastAPI.

## Setup

### Windows (recommended) — web workspace

1. Set **`GROQ_API_KEY`** (see [Environment variables](#environment-variables)).
2. Double-click **`run.bat`** in the repository root (or run `.\run.bat` in PowerShell).

It creates `venv` if needed, installs Python packages, starts the server in a **new console window**, waits until **http://127.0.0.1:8000** responds, then opens your **browser** to the workspace. Close the **VLA Server** window to stop.

### Manual venv (optional)

1. Open a PowerShell prompt in the repository root.
2. Run:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Start the server:

   ```powershell
   python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
   ```

4. Open **http://127.0.0.1:8000** in your browser.

Make sure MuJoCo is installed and any required license file is available.

In the workspace: drag colored objects on the table; use the chat panel for natural-language commands; watch execution output in the terminal panel.

## Running the CLI controller (no browser)

From the repository root, run:

```powershell
python franka_emika_panda\vla_controller.py
```

Then enter commands at the prompt, for example:

- `move to red box`
- `hover over blue box`
- `close gripper`
- `open gripper`
- `move home`
- `quit`

## Notes on configuration

- `franka_emika_panda/vla_controller.py` uses the OpenAI-compatible `openai` client.
- The script reads `GROQ_API_KEY` from the environment instead of storing it in code.
- Set `GROQ_API_KEY` locally before running the project.
- Optionally set `GROQ_BASE_URL` if your API endpoint differs from the default.
- Do not commit your API key into GitHub.

## Environment variables

Create a `.env` file or set these variables in your shell:

```powershell
setx GROQ_API_KEY "your_groq_api_key_here"
setx GROQ_BASE_URL "https://api.groq.com/openai/v1"
```

On PowerShell for the current session only:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
$env:GROQ_BASE_URL = "https://api.groq.com/openai/v1"
```

If you prefer a file-based example, copy `.env.example` to `.env` and fill in your key.

The project loads `.env` from the **repository root**. Non-empty values in `.env` override existing environment variables (so a real key in `.env` is used even if Windows has an empty `GROQ_API_KEY`).

## How it works

**Web workspace:** FastAPI runs a MuJoCo simulation thread and streams JPEG frames plus logs over a WebSocket; the browser UI in `web/` connects to `/ws` and serves from the same origin on port 8000.

**CLI:** The script loads `vla_scene.xml` and creates a MuJoCo model and simulation data. A passive MuJoCo viewer is launched for visualization. It captures a rendered image, sends the image and the user command to an LLM API.

In both modes, the LLM returns a JSON action plan like:

```json
[
  {"action": "hover_over_object", "target_name": "red_box"},
  {"action": "move_to_object", "target_name": "red_box"},
  {"action": "close_gripper"}
]
```

The controller then executes the plan by moving the arm, hovering, and operating the gripper.

## Known limitations

- This is a prototype: the LLM prompt assumes a fixed scene with three objects.
- Object names must match the built-in target mapping.
- The inverse kinematics solver is simple and may not handle all complex motions.
- Real-time perception is limited to rendered scene snapshots.

## Additional information

For details about the Panda MuJoCo model and scene assets, see `franka_emika_panda/README.md`.

---

Happy experimenting with the VLA Panda prototype!