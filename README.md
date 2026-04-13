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

## Project structure

- `franka_emika_panda/`
  - `vla_controller.py` — main VLA controller script.
  - `vla_scene.xml` — MuJoCo scene with the Panda robot and objects.
  - `panda.xml`, `scene.xml`, `mjx_*.xml` — robot model and scene files.
  - `README.md` — model-specific documentation for the Panda MJCF/MJX assets.
- `requirements.txt` — Python dependencies.
- `run.bat` — Windows helper to create/activate a venv, install dependencies, and run the controller.
- `.gitignore` — excludes temporary files, virtual environment folders, MuJoCo keys, and logs.

## Requirements

- Python 3.10+.
- MuJoCo 2.3.3 or later.
- A valid MuJoCo license key installed locally (e.g. `mjkey.txt`).
- GPU or CPU support for MuJoCo depending on your local setup.

## Setup

### Windows (recommended)

1. Open a PowerShell prompt in the repository root.
2. Run:

   ```powershell
   .\run.bat
   ```

This will create a virtual environment in `venv/`, install the required packages, verify imports, and then launch the controller.

### Manual setup

1. Create and activate a Python virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Make sure MuJoCo is installed and your license file is available.

## Running the controller

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

## How it works

1. The script loads `vla_scene.xml` and creates a MuJoCo model and simulation data.
2. A passive MuJoCo viewer is launched for visualization.
3. The script captures a rendered image from the scene and saves it temporarily.
4. It sends the image and the user command to an LLM API.
5. The API returns a JSON action plan like:
   
   ```json
   [
     {"action": "hover_over_object", "target_name": "red_box"},
     {"action": "move_to_object", "target_name": "red_box"},
     {"action": "close_gripper"}
   ]
   ```

6. The controller executes the plan by moving the arm, hovering, and operating the gripper.

## Known limitations

- This is a prototype: the LLM prompt assumes a fixed scene with three objects.
- Object names must match the built-in target mapping.
- The inverse kinematics solver is simple and may not handle all complex motions.
- Real-time perception is limited to rendered scene snapshots.

## Additional information

For details about the Panda MuJoCo model and scene assets, see `franka_emika_panda/README.md`.

---

Happy experimenting with the VLA Panda prototype!