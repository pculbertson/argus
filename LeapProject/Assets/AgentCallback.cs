using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AgentCallback : Agent {
    private ArticulationBody hand;
    private Rigidbody cube;
    private Camera cam1;
    private Camera cam2;
    private Color cam1BackgroundColor;
    private Color cam2BackgroundColor;
    private Light lightSource;

    private int rotationalJoints = 16;

    // Start is called before the first frame update
    void Start() {
        hand = this.GetComponentInChildren<ArticulationBody>();
        cube = this.GetComponentInChildren<Rigidbody>();
        cam1 = this.GetComponentsInChildren<Camera>()[0];
        cam1.clearFlags = CameraClearFlags.SolidColor;
        cam2 = this.GetComponentsInChildren<Camera>()[1];
        cam2.clearFlags = CameraClearFlags.SolidColor;
        lightSource = this.GetComponentInChildren<Light>();
    }

    public override void OnEpisodeBegin() { /** do nothing **/ }

    public override void CollectObservations(VectorSensor sensor) { /** do nothing **/ }

    public override void OnActionReceived(ActionBuffers actionBuffers) {
        // there are a total of 2 * 10 + 7 + 16 = 43 actions
        // * each of the two cameras has 7 DOFs and 3 exposed colors
        // * the cube has 7 DOFs
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

        // set the cube states
        cube.transform.localPosition = new Vector3(actionList[20], actionList[21], actionList[22]);
        Vector4 quat_cube = new Vector4(actionList[23], actionList[24], actionList[25], actionList[26]);
        quat_cube.Normalize();
        cube.transform.localRotation = new Quaternion(quat_cube[0], quat_cube[1], quat_cube[2], quat_cube[3]);

        // set the light source pose
        lightSource.transform.localPosition = new Vector3(actionList[27], actionList[28], actionList[29]);
        Vector4 quat_light = new Vector4(actionList[30], actionList[31], actionList[32], actionList[33]);
        quat_light.Normalize();
        lightSource.transform.localRotation = new Quaternion(quat_light[0], quat_light[1], quat_light[2], quat_light[3]);

        // set the hand states
        // WARNING: the order of the hand joints is breadth-first in the finger order middle, thumb, ring, index!
        // For instance, the first 4 joints are the base joints of the middle, thumb, ring, and index fingers. Then,
        // the next 4 joints are the medial joints in the same order, and so on.
        hand.SetJointPositions(actionList.GetRange(33, rotationalJoints)); // Set joint positions

        // concluding
        SetReward(1f); // arbitrary unused reward
        EndEpisode();  // let each episode be 1 action for simplicity
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        // heuristic used entirely for debugging
        var continuousActions = actionsOut.ContinuousActions;
        // for (int ii = 0; ii < continuousActions.Length; ii++) {
        //     // if colors, only generate numbers in the [0, 1] range
        //     if (ii == 7 || ii == 8 || ii == 9 || ii == 17 || ii == 18 || ii == 19) {
        //         continuousActions[ii] = Random.Range(0.0f, 1.0f);
        //     } else {
        //         continuousActions[ii] = Random.Range(-0.3f, 0.3f);
        //     }
        // }

        // debug: check joint ordering
        for (int ii = 0; ii < continuousActions.Length; ii++) {
            // if (ii == 37) {
            //     continuousActions[ii] = 1.57f;
            // } else {
            //     continuousActions[ii] = 0.0f;
            // }
            // Print each joint name
            Debug.Log(hand.jointName);
        }
    }
}
