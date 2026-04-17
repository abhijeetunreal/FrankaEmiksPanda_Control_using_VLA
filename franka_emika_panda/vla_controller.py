import mujoco
import mujoco.viewer
import cv2
import time
import json
import numpy as np
import os
import re
import base64
from openai import OpenAI

# ---------------------------------------------------------
# 1. Configuration (Using GROQ)
# ---------------------------------------------------------

def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(root_dir, ".env"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY environment variable. Set it before running the script."
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL
)

system_instruction = """
You are a robotic Vision-Language-Action (VLA) controller.
Analyze the provided image of the robot workspace and the user's natural language command.
Break the user's command down into a sequence of actionable primitives.

Manipulation actions:
1. "move_to_object": requires a "target_name" (e.g., "red_box", "blue_box", "green_box").
2. "hover_over_object": moves above the object, requires "target_name".
3. "close_gripper": grasps the object.
4. "open_gripper": releases the object.
5. "move_home": moves the arm back to a neutral ready pose (same as pre-emote home).

Fun / emote actions (no target_name; the arm has no separate head — nod/shake are end-effector motions):
6. "emote_nod": nod "yes" (vertical motion at the ready pose).
7. "emote_shake_no": shake side-to-side for "no".
8. "emote_wave": friendly side wave.
9. "emote_dance": short celebratory motion sequence.
10. "emote_clap": rapid gripper open/close like clapping (uses ready pose height).
11. "emote_yes": one clear affirmative nod (shorter than emote_nod).
12. "emote_no": short "no" shake (shorter than emote_shake_no).
13. "emote_rotate_wrist": spin the wrist (joint 7) at the ready pose.
14. "emote_bow": single smooth bow (lower and rise).
15. "emote_celebrate": brief raise upward then return (handled as part of the motion).

Object names in this scene:
- red_box (red cube)
- blue_box (blue cube)
- green_box (green sphere)

For commands like nod, dance, clap, say yes/no, wave, bow, celebrate, or rotate wrist, output the matching emote_* action(s). Each emote is automatically run from the ready pose and the arm returns to ready afterward — you do not need to insert extra move_home before/after emotes unless mixing with manipulation.

Output ONLY a valid JSON array of action objects. Do not include any conversational text.
Example grasp: [{"action": "hover_over_object", "target_name": "red_box"}, {"action": "move_to_object", "target_name": "red_box"}, {"action": "close_gripper"}]
Example emote: [{"action": "emote_dance"}]
"""

TARGET_NAME_MAP = {
    "red_box": "red_box",
    "red box": "red_box",
    "red cube": "red_box",
    "red": "red_box",
    "blue_box": "blue_box",
    "blue box": "blue_box",
    "blue cube": "blue_box",
    "blue": "blue_box",
    "green_box": "green_box",
    "green box": "green_box",
    "green sphere": "green_box",
    "green ball": "green_box",
    "green": "green_box",
}
VALID_TARGETS = set(TARGET_NAME_MAP.values())

FUN_ACTIONS = frozenset(
    {
        "emote_nod",
        "emote_shake_no",
        "emote_wave",
        "emote_dance",
        "emote_clap",
        "emote_yes",
        "emote_no",
        "emote_rotate_wrist",
        "emote_bow",
        "emote_celebrate",
    }
)

KNOWN_PLAN_ACTIONS = frozenset(
    {
        "move_to_object",
        "hover_over_object",
        "close_gripper",
        "open_gripper",
        "move_home",
    }
) | FUN_ACTIONS

# Chained pick/place steps stay at the workspace; do not return home between these.
MID_MANIPULATION_ACTIONS = frozenset(
    {
        "move_to_object",
        "hover_over_object",
        "open_gripper",
        "close_gripper",
    }
)


def _skip_ensure_home_before_step(prev_action, action):
    """Avoid breaking grasp sequences: only skip when continuing mid-manipulation."""
    if prev_action is None:
        return False
    return prev_action in MID_MANIPULATION_ACTIONS and action in MID_MANIPULATION_ACTIONS


# ---------------------------------------------------------
# 2. Inverse Kinematics (IK) Solver
# ---------------------------------------------------------
def solve_ik(target_pos, m, d, viewer, steps=300):
    """Translates 3D Cartesian coordinates to 7-DOF joint angles."""
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_TIP_NAME)
    use_site = ee_site_id != -1
    ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    for _ in range(steps):
        current_ee_pos = d.site_xpos[ee_site_id] if use_site else d.xpos[ee_body_id]
        error = target_pos - current_ee_pos
        
        if np.linalg.norm(error) < 0.01:
            break
            
        jacp = np.zeros((3, m.nv))
        if use_site:
            mujoco.mj_jacSite(m, d, jacp, None, ee_site_id)
        else:
            mujoco.mj_jacBodyCom(m, d, jacp, None, ee_body_id)
        
        lambda_sq = 0.01 
        J_pinv = jacp.T @ np.linalg.inv(jacp @ jacp.T + lambda_sq * np.eye(3))
        dq = J_pinv @ error * 0.5 
        
        for i in range(7):
            d.ctrl[i] += dq[i] * 0.05 
            
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.005)

# ---------------------------------------------------------
# 3. VLA Logic & Execution
# ---------------------------------------------------------
def encode_image(image_path):
    """Converts local image to base64 for the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def normalize_target_name(name):
    if not name:
        return None
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    return TARGET_NAME_MAP.get(key)


def get_target_from_command(command):
    if not command:
        return None
    text = command.lower()
    for alias, target in TARGET_NAME_MAP.items():
        if alias in text:
            return target
    return None


def sanitize_plan_targets(plan, user_command):
    command_target = get_target_from_command(user_command)
    for step in plan:
        action = step.get("action")
        if action in ["move_to_object", "hover_over_object"]:
            normalized = normalize_target_name(step.get("target_name"))
            if normalized:
                step["target_name"] = normalized
            elif command_target:
                step["target_name"] = command_target
    return plan


def get_gripper_position(d):
    if len(d.ctrl) <= 7:
        return GRIPPER_OPEN
    return float(np.mean(d.ctrl[7:]))


def is_gripper_open(d):
    return get_gripper_position(d) >= (GRIPPER_OPEN + GRIPPER_CLOSED) / 2


def plan_needs_open_before_grasp(plan, user_command):
    if not isinstance(plan, list):
        return False
    if not any(step.get("action") == "close_gripper" for step in plan):
        return False
    return any(step.get("action") in ["move_to_object", "hover_over_object"] for step in plan)


def sanitize_execution_plan(plan, user_command, d):
    command_target = get_target_from_command(user_command)
    sanitized = []
    last_step = None

    for step in plan:
        action = step.get("action")
        new_step = dict(step)

        if action in ["move_to_object", "hover_over_object"]:
            normalized = normalize_target_name(step.get("target_name"))
            new_step["target_name"] = normalized or command_target

        if last_step and action == last_step.get("action") and new_step.get("target_name") == last_step.get("target_name"):
            continue
        if action in ["open_gripper", "close_gripper"] and last_step and last_step.get("action") == action:
            continue

        sanitized.append(new_step)
        last_step = new_step

    if plan_needs_open_before_grasp(sanitized, user_command) and not is_gripper_open(d):
        already_open_before_move = False
        for step in sanitized:
            if step.get("action") == "open_gripper":
                already_open_before_move = True
                break
            if step.get("action") in ["move_to_object", "hover_over_object"]:
                break

        if not already_open_before_move:
            insert_index = 0
            for i, step in enumerate(sanitized):
                if step.get("action") in ["move_to_object", "hover_over_object"]:
                    insert_index = i
                    break
            sanitized.insert(insert_index, {"action": "open_gripper"})

    optimized = []
    for i, step in enumerate(sanitized):
        if step.get("action") == "hover_over_object" and i + 1 < len(sanitized):
            next_step = sanitized[i + 1]
            if next_step.get("action") == "move_to_object" and step.get("target_name") == next_step.get("target_name"):
                continue
        optimized.append(step)

    filtered = []
    for step in optimized:
        act = step.get("action")
        if act in KNOWN_PLAN_ACTIONS:
            filtered.append(step)
        else:
            print(f"[Sanitize] Dropping unknown action: {act!r}")

    return filtered


def get_vla_plan(image_path, user_command):
    print("\n[VLA] Processing image with Groq (Llama 3.2 Vision)...")
    
    base64_image = encode_image(image_path)
    
    try:
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # <--- New, active Llama 4 Vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_instruction}\n\nUser Command: {user_command}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0
        )
        
        # Parse the JSON response safely
        raw_text = response.choices[0].message.content.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"[VLA Error]: {e}")
        return []

GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 255.0
GRIPPER_MOVE_STEPS = 200
HOVER_HEIGHT = 0.22  # keep the end effector safely above object top
APPROACH_HEIGHT = 0.18  # height above object center before lowering for grasp
GRIPPER_CLEARANCE = 0.065  # extra clearance to avoid table collisions
OBJECT_TOP_OFFSET = 0.02  # half-height for cubes in the scene
GRAB_HEIGHT_OFFSET = -0.005  # lower toward the cube mid-height for a better grip
GRIPPER_TIP_NAME = "gripper_tip"

READY_EE_POS = np.array([0.4, 0.0, 0.7])
# Matches main() initial qpos[6]; used to undo redundant wrist (joint7) after emote_rotate_wrist.
HOME_JOINT7 = 0.785
READY_EE_DISTANCE_TOL = 0.045
EMOTE_SETTLE_STEPS = 25
EMOTE_Z_MIN = 0.38


def get_ee_position(m, d):
    ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_TIP_NAME)
    if ee_site_id != -1:
        return d.site_xpos[ee_site_id].copy()
    ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return d.xpos[ee_body_id].copy()


def is_near_ready_pose(m, d):
    return float(np.linalg.norm(get_ee_position(m, d) - READY_EE_POS)) < READY_EE_DISTANCE_TOL


def settle_physics(m, d, viewer, steps=EMOTE_SETTLE_STEPS):
    for _ in range(steps):
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.005)


def ensure_ready_pose(m, d, viewer):
    if is_near_ready_pose(m, d):
        return
    solve_ik(READY_EE_POS.copy(), m, d, viewer, steps=350)
    settle_physics(m, d, viewer)


def return_to_ready_pose(m, d, viewer):
    """IK to ready EE position, then reset wrist joint7 (position-only IK cannot fix redundant angle)."""
    solve_ik(READY_EE_POS.copy(), m, d, viewer, steps=350)
    slew_joint7_to(m, d, viewer, HOME_JOINT7, interp_steps=45, hold_each=3)
    solve_ik(READY_EE_POS.copy(), m, d, viewer, steps=200)
    settle_physics(m, d, viewer)


def go_to_home_pose(m, d, viewer):
    """Always run IK to ready pose (no distance shortcut). Use before executing a plan."""
    return_to_ready_pose(m, d, viewer)


def emote_target_from_offset(dx, dy, dz):
    target = READY_EE_POS.copy() + np.array([dx, dy, dz], dtype=float)
    target[2] = max(target[2], EMOTE_Z_MIN)
    return target


def clamp_joint7(m, angle):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    if jid == -1:
        return float(np.clip(angle, -2.8973, 2.8973))
    lo, hi = m.jnt_range[jid]
    return float(np.clip(angle, lo, hi))


def slew_joint7_to(m, d, viewer, target_angle, interp_steps=40, hold_each=4):
    target_angle = clamp_joint7(m, target_angle)
    start = float(d.ctrl[6])
    for k in range(1, interp_steps + 1):
        alpha = k / interp_steps
        d.ctrl[6] = start + alpha * (target_angle - start)
        for _ in range(hold_each):
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.003)


def set_gripper_quick(ctrl_value, m, d, viewer, steps=40):
    for i in range(7, len(d.ctrl)):
        d.ctrl[i] = ctrl_value
    for _ in range(steps):
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.003)


def run_emote(action, m, d, viewer):
    if action == "emote_nod":
        _emote_nod(m, d, viewer, cycles=4)
    elif action == "emote_yes":
        _emote_nod(m, d, viewer, cycles=2)
    elif action in ("emote_shake_no", "emote_no"):
        _emote_shake(m, d, viewer, cycles=5 if action == "emote_shake_no" else 3)
    elif action == "emote_wave":
        _emote_wave(m, d, viewer)
    elif action == "emote_dance":
        _emote_dance(m, d, viewer)
    elif action == "emote_clap":
        _emote_clap(m, d, viewer)
    elif action == "emote_rotate_wrist":
        _emote_rotate_wrist(m, d, viewer)
    elif action == "emote_bow":
        _emote_bow(m, d, viewer)
    elif action == "emote_celebrate":
        _emote_celebrate(m, d, viewer)


def _emote_nod(m, d, viewer, cycles=4):
    for _ in range(cycles):
        solve_ik(emote_target_from_offset(0, 0, -0.07), m, d, viewer, steps=120)
        solve_ik(emote_target_from_offset(0, 0, 0.05), m, d, viewer, steps=120)


def _emote_shake(m, d, viewer, cycles=5):
    for _ in range(cycles):
        solve_ik(emote_target_from_offset(0, 0.09, 0), m, d, viewer, steps=100)
        solve_ik(emote_target_from_offset(0, -0.09, 0), m, d, viewer, steps=100)


def _emote_wave(m, d, viewer):
    base_j7 = float(d.ctrl[6])
    for _ in range(4):
        solve_ik(emote_target_from_offset(0, 0.1, 0.04), m, d, viewer, steps=90)
        slew_joint7_to(m, d, viewer, base_j7 + 0.45, interp_steps=18, hold_each=3)
        solve_ik(emote_target_from_offset(0, -0.02, 0.04), m, d, viewer, steps=90)
        slew_joint7_to(m, d, viewer, base_j7 - 0.35, interp_steps=18, hold_each=3)
    slew_joint7_to(m, d, viewer, base_j7, interp_steps=22, hold_each=3)


def _emote_dance(m, d, viewer):
    seq = [
        (0.0, 0.1, 0.05),
        (0.0, -0.1, 0.0),
        (0.05, 0.0, 0.06),
        (-0.05, 0.0, 0.0),
        (0.0, 0.08, -0.04),
        (0.0, -0.08, 0.03),
    ]
    for dx, dy, dz in seq:
        solve_ik(emote_target_from_offset(dx, dy, dz), m, d, viewer, steps=100)
    j0 = float(d.ctrl[6])
    slew_joint7_to(m, d, viewer, j0 + 1.2, interp_steps=25, hold_each=2)
    slew_joint7_to(m, d, viewer, j0 - 0.8, interp_steps=25, hold_each=2)
    slew_joint7_to(m, d, viewer, j0, interp_steps=20, hold_each=2)


def _emote_clap(m, d, viewer):
    open_gripper(m, d, viewer)
    solve_ik(READY_EE_POS.copy(), m, d, viewer, steps=200)
    for _ in range(5):
        set_gripper_quick(GRIPPER_CLOSED, m, d, viewer, steps=28)
        set_gripper_quick(GRIPPER_OPEN, m, d, viewer, steps=28)
    open_gripper(m, d, viewer)


def _emote_rotate_wrist(m, d, viewer):
    solve_ik(READY_EE_POS.copy(), m, d, viewer, steps=180)
    start = float(d.ctrl[6])
    target = start + 2.0 * np.pi
    slew_joint7_to(m, d, viewer, target, interp_steps=70, hold_each=3)
    settle_physics(m, d, viewer, steps=15)


def _emote_bow(m, d, viewer):
    solve_ik(emote_target_from_offset(0.06, 0, -0.14), m, d, viewer, steps=200)
    settle_physics(m, d, viewer, steps=18)
    solve_ik(READY_EE_POS.copy(), m, d, viewer, steps=220)


def _emote_celebrate(m, d, viewer):
    up = READY_EE_POS.copy() + np.array([0.0, 0.0, 0.16])
    up[2] = max(up[2], EMOTE_Z_MIN)
    solve_ik(up, m, d, viewer, steps=220)
    settle_physics(m, d, viewer, steps=15)
    wiggle = float(d.ctrl[6])
    slew_joint7_to(m, d, viewer, wiggle + 0.6, interp_steps=15, hold_each=3)
    slew_joint7_to(m, d, viewer, wiggle - 0.6, interp_steps=15, hold_each=3)


def set_gripper(ctrl_value, m, d, viewer, steps=GRIPPER_MOVE_STEPS):
    for i in range(7, len(d.ctrl)):
        d.ctrl[i] = ctrl_value
    for _ in range(steps):
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.005)


def open_gripper(m, d, viewer):
    print("[Execution] Opening gripper")
    set_gripper(GRIPPER_OPEN, m, d, viewer)


def close_gripper(m, d, viewer):
    print("[Execution] Closing gripper")
    set_gripper(GRIPPER_CLOSED, m, d, viewer)


def hover_over_object(target_pos, m, d, viewer):
    hover_pos = target_pos.copy()
    hover_pos[2] += OBJECT_TOP_OFFSET + HOVER_HEIGHT + GRIPPER_CLEARANCE
    solve_ik(hover_pos, m, d, viewer)


def approach_object(target_pos, m, d, viewer):
    approach_pos = target_pos.copy()
    approach_pos[2] += OBJECT_TOP_OFFSET + APPROACH_HEIGHT + GRIPPER_CLEARANCE
    solve_ik(approach_pos, m, d, viewer)

    lower_pos = target_pos.copy()
    lower_pos[2] += OBJECT_TOP_OFFSET + GRAB_HEIGHT_OFFSET
    solve_ik(lower_pos, m, d, viewer)

    # settle the final pose after the last descent so the gripper does not bounce back
    for _ in range(20):
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.005)


def execute_plan(plan, m, d, viewer, user_command=None):
    print("\n[Execution] Executing VLA Plan:")
    command_target = get_target_from_command(user_command)
    if is_near_ready_pose(m, d):
        print("[Execution] Near home; locking in exact ready pose before plan steps.")
    else:
        print("[Execution] Not at home; moving to ready pose before plan steps.")
    go_to_home_pose(m, d, viewer)
    prev_action = None
    for step in plan:
        action = step.get("action")
        target = normalize_target_name(step.get("target_name"))
        if not target and command_target and action in ["move_to_object", "hover_over_object"]:
            print(f"[Execution] Overriding invalid target with command target: {command_target}")
            target = command_target
            step["target_name"] = target

        print(f" -> {action} | Target: {target}")

        if action in FUN_ACTIONS:
            ensure_ready_pose(m, d, viewer)
            run_emote(action, m, d, viewer)
            return_to_ready_pose(m, d, viewer)
            prev_action = action
            continue

        if not _skip_ensure_home_before_step(prev_action, action):
            ensure_ready_pose(m, d, viewer)

        if action in ["move_to_object", "hover_over_object"]:
            if target:
                obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
                if obj_id == -1:
                    print(f"[Execution] Unknown target body '{target}', skipping action")
                    prev_action = action
                    continue
                target_pos = d.xpos[obj_id].copy()
                
                if action == "move_to_object":
                    approach_object(target_pos, m, d, viewer)
                elif action == "hover_over_object":
                    hover_over_object(target_pos, m, d, viewer)
                
        elif action == "move_home":
            solve_ik(READY_EE_POS.copy(), m, d, viewer)
            
        elif action == "close_gripper":
            close_gripper(m, d, viewer)
                
        elif action == "open_gripper":
            open_gripper(m, d, viewer)

        prev_action = action

# ---------------------------------------------------------
# 4. Main Loop
# ---------------------------------------------------------
def main():
    print("Loading Franka Panda Environment...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, 'vla_scene.xml')
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    d.qpos[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    d.ctrl[:7] = d.qpos[:7]
    for i in range(7, len(d.ctrl)):
        d.ctrl[i] = GRIPPER_OPEN
    mujoco.mj_forward(m, d)
    
    renderer = mujoco.Renderer(m, 480, 640)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        for _ in range(200):
            mujoco.mj_step(m, d); viewer.sync()
            
        while True:
            cmd = input("\nEnter robot command (or 'quit'): ")
            if cmd.lower() in ['quit', 'exit']: break
                
            renderer.update_scene(d, camera="main_cam")
            image_path = os.path.join(script_dir, "current_state.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            
            plan = get_vla_plan(image_path, cmd)
            if plan:
                plan = sanitize_execution_plan(plan, cmd, d)
                print(json.dumps(plan, indent=2))
                execute_plan(plan, m, d, viewer, cmd)
            else:
                print("[!] No actionable plan generated.")
                
            if os.path.exists(image_path): os.remove(image_path)

if __name__ == "__main__":
    main()