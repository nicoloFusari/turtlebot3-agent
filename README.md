# TurtleBot3 Agent

An agent you give commands to in natural language, which drives a TurtleBot3 Burger around the house, looks for objects and people, and reports on what it finds.

> "Go to the bathroom and tell me if there's toilet paper"
> "Is Vittorio home?"
> "Where is Nicolò's backpack?"

The agent chooses where to go on the map, navigates with Nav2, spins in place looking with the camera, and confirms what it saw before answering. It also recognizes the housemates' faces and who some objects belong to, with a photo memory that stays on the PC.

## How it works

| Part | What it does |
| --- | --- |
| Agent (`agent.py`) | LangChain agent with four tools: `get_map`, `where_am_i`, `go_to_location`, `inspect`. It decides which rooms and points to visit and in what order. |
| Navigation (`ros_bridge.py`) | Sends goals to Nav2 (`NavigateToPose`) and performs the inspection spin in 20° steps with the `Spin` behavior. It is the only file that uses ROS 2. |
| Detector (`detector.py`) | Open-vocabulary YOLOE with segmentation: looks for the requested object, with the label in English, in the camera frames. |
| Visual verification | When the detector finds something, a multimodal model looks at the frame and confirms or rejects it. |
| Memory (`memory.py`) | Reference photos of faces and objects: faces with OpenCV's YuNet and SFace, objects with CLIP ViT-B/32 on the masked crop. It tells who what the robot found belongs to. |

Other behaviors:

- **People:** for `person` the detector is face recognition, without visual verification. It also works while the robot is navigating: if it sees the face of the person it is looking for, it stops there.
- **Targeted search:** with `who` (e.g. "Vitto's backpack") the spin continues until it finds the right one, and also reports the ones belonging to others that it saw.
- **Sound signal:** when it finds what it was looking for, the robot plays a melody with the OpenCR buzzer.

## Requirements

**Robot**
- TurtleBot3 Burger with ROS 2 bringup running (`turtlebot3_node`, LiDAR, `robot_state_publisher`).
- A camera publishing `sensor_msgs/CompressedImage` on `/camera/image_raw/compressed`.

**PC**
- Ubuntu 24.04, ROS 2 Jazzy, Nav2, `turtlebot3` and `turtlebot3_msgs` packages.
- Python 3.12.
- An OpenAI API key. The models are set in `agent.py` and are changed with `init_chat_model`.
- About 1 GB of space for the model weights, downloaded on first use into `models/`.

Runs on CPU. On the PC used for development YOLOE analyzes about 3–4 frames per second at 640 px, face recognition about 25.



### 1. Agent identity and limits: `costituzione.md`

The agent reads it at startup and adds it to its instructions, and **without this file it does not start**. It is excluded from the repository because it contains personal information. A minimal example:


### 2. Map and places: `maps/`, `places.yaml`

`maps/` contains the Nav2 map. `places.yaml` gives names to points on the map: for each room a pose, some viewpoints and the connections between rooms.

```yaml
frame_id: map
map: map_v2.yaml
places:
  corridoio:
    pose: {x: 3.726, y: -1.042, yaw: -1.491}
    viewpoints:
      camera_vitto: {x: 3.594, y: -2.126, yaw: 2.951}
connections:
  - {a: home, b: corridoio}
```

The coordinates are valid only for the specified map. For another house you need a new map (SLAM with Cartographer or slam_toolbox) and new poses: bring the robot to the point and read its position and orientation with `ros2 topic echo /amcl_pose --once` (the yaw is derived from the quaternion).

Viewpoint names matter: the agent uses them to figure out where to search (e.g. `tavolo_cucina` for kitchen items).


## Files

| File | Contents |
| --- | --- |
| `agent.py` | Agent, tools, prompt and visual verification |
| `ros_bridge.py` | Bridge to ROS 2: Nav2, inspection spin, camera, buzzer |
| `detector.py` | Real-time YOLOE detector |
| `memory.py` | Memory of faces and objects, with the commands to manage it |
| `places.yaml` | Rooms, viewpoints and connections |
| `maps/` | Nav2 map |

## Notes on Nav2

With this robot and a home Wi-Fi network some changes to the Nav2 parameters (`burger.yaml`) were needed, which are not in this repository.
