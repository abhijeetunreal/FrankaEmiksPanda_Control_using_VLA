import mujoco
import mujoco.viewer
import cv2
import time
import json
import numpy as np
import os
import base64
from openai import OpenAI

# ---------------------------------------------------------
# 1. Configuration (Using GROQ)
# ---------------------------------------------------------
# SECURITY WARNING: Delete this key from console.groq.com after testing!
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
)

system_instruction = """
You are a robotic Vision-Language-Action (VLA) controller.
Analyze the provided image of the robot workspace and the user's natural language command.
Break the user's command down into a sequence of actionable primitives. 

Available actions:
1. "move_to_object": requires a "target_name" (e.g., "red_box", "blue_box", "green_box").
2. "hover_over_object": moves 15cm above the object, requires "target_name".
3. "close_gripper": grasps the object.
4. "open_gripper": releases the object.
5. "move_home": moves the arm back to a neutral position.

Object names in this scene:
- red_box (red cube)
- blue_box (blue cube)
- green_box (green sphere)

Output ONLY a valid JSON array of action objects. Do not include any conversational text.
Example: [{"action": "hover_over_object", "target_name": "red_box"}, {"action": "move_to_object", "target_name": "red_box"}, {"action": "close_gripper"}]
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
    for step in plan:
        action = step.get("action")
        target = normalize_target_name(step.get("target_name"))
        if not target and command_target and action in ["move_to_object", "hover_over_object"]:
            print(f"[Execution] Overriding invalid target with command target: {command_target}")
            target = command_target
            step["target_name"] = target

        print(f" -> {action} | Target: {target}")
        
        if action in ["move_to_object", "hover_over_object"]:
            if target:
                obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
                if obj_id == -1:
                    print(f"[Execution] Unknown target body '{target}', skipping action")
                    continue
                target_pos = d.xpos[obj_id].copy()
                
                if action == "move_to_object":
                    approach_object(target_pos, m, d, viewer)
                elif action == "hover_over_object":
                    hover_over_object(target_pos, m, d, viewer)
                
        elif action == "move_home":
            solve_ik(np.array([0.4, 0.0, 0.7]), m, d, viewer)
            
        elif action == "close_gripper":
            close_gripper(m, d, viewer)
                
        elif action == "open_gripper":
            open_gripper(m, d, viewer)

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
                print(json.dumps(plan, indent=2))
                execute_plan(plan, m, d, viewer, cmd)
            else:
                print("[!] No actionable plan generated.")
                
            if os.path.exists(image_path): os.remove(image_path)

if __name__ == "__main__":
    main()