using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AgentCallback : Agent {
    private ArticulationBody hand;
    private Dictionary<int, int> jointMap;  // mjpc joint index -> unity joint index
    private Rigidbody cube;
    private Camera cam1;
    private Camera cam2;
    private Camera depthCam1;
    private Camera depthCam2;
    private Color cam1BackgroundColor;
    private Color cam2BackgroundColor;
    private Color depthCam1BackgroundColor; // doesn't affect background for depthCam (placeholder)
    private Color depthCam2BackgroundColor; // doesn't affect background for depthCam (placeholder)
    private Light lightSource;

    // Start is called before the first frame update
    void Start() {
        hand = this.GetComponentInChildren<ArticulationBody>();
        cube = this.GetComponentInChildren<Rigidbody>();
        lightSource = this.GetComponentInChildren<Light>();


        Camera[] cameras = this.GetComponentsInChildren<Camera>();
        
        foreach (Camera cam in cameras)
        {
            switch (cam.name)
            {
                case "cam1":
                    cam1 = cam;
                    cam1.clearFlags = CameraClearFlags.SolidColor;
                    break;
                case "cam2":
                    cam2 = cam;
                    cam2.clearFlags = CameraClearFlags.SolidColor;
                    break;
                case "depthCam1":
                    depthCam1 = cam;
                    depthCam1.clearFlags = CameraClearFlags.SolidColor;
                    break;
                case "depthCam2":
                    depthCam2 = cam;
                    depthCam2.clearFlags = CameraClearFlags.SolidColor;
                    break;
                default:
                    Debug.LogWarning("Camera with unexpected name found: " + cam.name);
                    break;
            }
        }        

        // setting up the dictionary for desired joint order
        jointMap = new Dictionary<int, int>();
        ArticulationBody[] allJoints = GetComponentsInChildren<ArticulationBody>();
        List<string> jointNames = new List<string>() {
            "mcp_joint", "pip", "dip", "fingertip",
            "mcp_joint_2", "pip_2", "dip_2", "fingertip_2",
            "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
            "pip_4", "thumb_pip", "thumb_dip", "thumb_fingertip",
        };  // order of joints in mjpc by name
        foreach (var joint in allJoints) {
            if (jointNames.Contains(joint.name)) {
                int idx = jointNames.IndexOf(joint.name);  // get index in jointNames that matches joint.name
                jointMap[idx] = joint.index - 2;  // the mount and palm joints are not counted, come first
            }
        }
    }

    public override void OnEpisodeBegin() { /** do nothing **/ }

    public override void CollectObservations(VectorSensor sensor) { /** do nothing **/ }

    public override void OnActionReceived(ActionBuffers actionBuffers) {
        // there are a total of 3 * 10 + 7 + 7 + 16 = 60 actions
        // * each of the two cameras has 7 DOFs and 3 exposed colors
        // The exposed colors don't matter for depth due to shader configuration
        // * the cube has 7 DOFs
        // * the light pose has 7 DOFs
        // * the hand has 16 DOFs
        // we assume that poses are in the order (x, y, z, qx, qy, qz, qw), which is Unity's convention.

        // retrieve all actions into a list
        var continuousActions = actionBuffers.ContinuousActions;
        List<float> actionList = new List<float>(continuousActions.Length);
        for (int ii = 0; ii < continuousActions.Length; ii++) {
            actionList.Add(continuousActions[ii]);
        }

        // set the camera states
        cam1.transform.localPosition = new Vector3(actionList[0], actionList[1], actionList[2]);
        Vector4 quat_cam1 = new Vector4(actionList[3], actionList[4], actionList[5], actionList[6]);
        quat_cam1.Normalize();
        cam1.transform.localRotation = new Quaternion(quat_cam1[0], quat_cam1[1], quat_cam1[2], quat_cam1[3]);
        cam1BackgroundColor.r = actionList[7];
        cam1BackgroundColor.g = actionList[8];
        cam1BackgroundColor.b = actionList[9];
        cam1.backgroundColor = cam1BackgroundColor;

        cam2.transform.localPosition = new Vector3(actionList[10], actionList[11], actionList[12]);
        Vector4 quat_cam2 = new Vector4(actionList[13], actionList[14], actionList[15], actionList[16]);
        quat_cam2.Normalize();
        cam2.transform.localRotation = new Quaternion(quat_cam2[0], quat_cam2[1], quat_cam2[2], quat_cam2[3]);
        cam2BackgroundColor.r = actionList[17];
        cam2BackgroundColor.g = actionList[18];
        cam2BackgroundColor.b = actionList[19];
        cam2.backgroundColor = cam2BackgroundColor;

        depthCam1.transform.localPosition = new Vector3(actionList[20], actionList[21], actionList[22]);
        Vector4 quat_depth1_cam = new Vector4(actionList[23], actionList[24], actionList[25], actionList[26]);
        quat_depth1_cam.Normalize();
        depthCam1.transform.localRotation = new Quaternion(quat_depth1_cam[0], quat_depth1_cam[1], quat_depth1_cam[2], quat_depth1_cam[3]);
        // Initialize, but don't use background values for depth (Placeholders)
        depthCam1BackgroundColor.r = actionList[27];
        depthCam1BackgroundColor.g = actionList[28];
        depthCam1BackgroundColor.b = actionList[29];
        depthCam1.backgroundColor = Color.white;       

        depthCam2.transform.localPosition = new Vector3(actionList[30], actionList[31], actionList[32]);
        Vector4 quat_depth2_cam = new Vector4(actionList[33], actionList[34], actionList[35], actionList[36]);
        quat_depth2_cam.Normalize();
        depthCam2.transform.localRotation = new Quaternion(quat_depth2_cam[0], quat_depth2_cam[1], quat_depth2_cam[2], quat_depth2_cam[3]);
        // Initialize, but don't use background values for depth (Placeholders)
        depthCam2BackgroundColor.r = actionList[37];
        depthCam2BackgroundColor.g = actionList[38];
        depthCam2BackgroundColor.b = actionList[39];
        depthCam2.backgroundColor = Color.white; 

        // set the cube states
        cube.transform.localPosition = new Vector3(actionList[40], actionList[41], actionList[42]);
        Vector4 quat_cube = new Vector4(actionList[43], actionList[44], actionList[45], actionList[46]);
        quat_cube.Normalize();
        cube.transform.localRotation = new Quaternion(quat_cube[0], quat_cube[1], quat_cube[2], quat_cube[3]);

        // set the light source pose
        lightSource.transform.localPosition = new Vector3(actionList[47], actionList[48], actionList[49]);
        Vector4 quat_light = new Vector4(actionList[50], actionList[51], actionList[52], actionList[53]);
        quat_light.Normalize();

        // Set the light source to look at the cube
        Vector3 lookAt = cube.transform.position - lightSource.transform.position;
        lightSource.transform.rotation = Quaternion.LookRotation(lookAt);

        // Take the action as a delta from this current rotation
        Quaternion delta_rotation = new Quaternion(quat_light[0], quat_light[1], quat_light[2], quat_light[3]);
        lightSource.transform.rotation = delta_rotation * lightSource.transform.rotation;

        // set the hand states
        var jointPositions = new float[16];
        foreach (var pair in jointMap) {
            jointPositions[pair.Value] = actionList[54 + pair.Key];
        }

        hand.SetJointPositions(jointPositions.ToList()); // Set joint positions

        // concluding
        SetReward(1f); // arbitrary unused reward
        EndEpisode();  // let each episode be 1 action for simplicity
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        // heuristic used entirely for debugging
        var continuousActions = actionsOut.ContinuousActions;
        for (int ii = 0; ii < continuousActions.Length; ii++) {
            // if colors, only generate numbers in the [0, 1] range
            if (ii == 7 || ii == 8 || ii == 9 || ii == 17 || ii == 18 || ii == 19 ||
                 ii == 27 || ii == 28 || ii == 29 || ii == 37 || ii == 38 || ii == 39) {
                    // retrieve color for depth background but don't use them 
                continuousActions[ii] = Random.Range(0.0f, 1.0f);
            } else {
                continuousActions[ii] = Random.Range(-0.3f, 0.3f);
            }
        }

        // debug: check joint ordering
        // for (int ii = 0; ii < continuousActions.Length; ii++) {
        //     if (ii == 34) {  // this sets the wrong finger!!!
        //         continuousActions[ii] = 1.57f;
        //     } else {
        //         continuousActions[ii] = 0.0f;
        //     }
        //     // Print each joint name
        //     // Debug.Log(hand.jointName);
        // }
    }
}
