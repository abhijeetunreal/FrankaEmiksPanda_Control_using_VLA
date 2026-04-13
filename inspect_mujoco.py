import mujoco, os
xml_path = os.path.join('franka_emika_panda', 'vla_scene.xml')
m = mujoco.MjModel.from_xml_path(xml_path)
print('nu', m.nu, 'nv', m.nv, 'na', m.na, 'nctrl', m.nctrl)
print('actuator_ctrlrange[:10]=', list(m.actuator_ctrlrange[:10]))
print('actuator_names:', [m.actuator_id2name(i) for i in range(min(m.nu, 20))])
