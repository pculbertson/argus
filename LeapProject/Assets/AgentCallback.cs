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

    private int rotationalJoints = 16;

    // Start is called before the first frame update
    void Start() {
        hand = this.GetComponentInChildren<ArticulationBody>();
        cube = this.GetComponentInChildren<Rigidbody>();
        cam1 = this.GetComponentsInChildren<Camera>()[0];
        cam2 = this.GetComponentsInChildren<Camera>()[1];
    }

    public override void OnEpisodeBegin() { /** do nothing **/ }

    public override void CollectObservations(VectorSensor sensor) { /** do nothing **/ }

    public override void OnActionReceived(ActionBuffers actionBuffers) {
        // there are a total of 2 * 7 + 7 + 16 = 37 actions
        // * each of the two cameras has 7 DOFs
        // * the cube has 7 DOFs
        // * the hand has 16 DOFs
        // we assume that poses are in the order (x, y, z, qx, qy, qz, qw). This is the OPPOSITE of the
        // observation convention!
        // the Quaternion object in Unity constructs quaternions using the (qx, qy, qz, qw) convention.

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
        cam1.transform.rotation = new Quaternion(quat_cam1[0], quat_cam1[1], quat_cam1[2], quat_cam1[3]);

        cam2.transform.localPosition = new Vector3(actionList[7], actionList[8], actionList[9]);
        Vector4 quat_cam2 = new Vector4(actionList[10], actionList[11], actionList[12], actionList[13]);
        quat_cam2.Normalize();
        cam2.transform.rotation = new Quaternion(quat_cam2[0], quat_cam2[1], quat_cam2[2], quat_cam2[3]);

        // set the cube states
        cube.transform.localPosition = new Vector3(actionList[14], actionList[15], actionList[16]);
        Vector4 quat_cube = new Vector4(actionList[17], actionList[18], actionList[19], actionList[20]);
        quat_cube.Normalize();
        cube.transform.rotation = new Quaternion(quat_cube[0], quat_cube[1], quat_cube[2], quat_cube[3]);

        // set the hand states
        hand.SetJointPositions(actionList.GetRange(21, rotationalJoints)); // Set joint positions

        // concluding
        SetReward(1f); // arbitrary unused reward
        EndEpisode();  // let each episode be 1 action for simplicity
    }
}
