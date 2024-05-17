using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AgentCallback : Agent
{
    private ArticulationBody hand;
    private Rigidbody cube;
    private Vector4 quat;
    private int rotationalJoints = 16;

    // Start is called before the first frame update
    void Start()
    {
        hand = this.GetComponentInChildren<ArticulationBody>();
        cube = GameObject.Find("cube").GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // do nothing
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // cube states
        sensor.AddObservation(cube.position.x);
        sensor.AddObservation(cube.position.y);
        sensor.AddObservation(cube.position.z);
        sensor.AddObservation(cube.rotation.w);
        sensor.AddObservation(cube.rotation.x);
        sensor.AddObservation(cube.rotation.y);
        sensor.AddObservation(cube.rotation.z);

        // hand states
        List<float> jointPositions = new List<float>();
        hand.GetJointPositions(jointPositions); // Correctly get joint positions
        for (int ii = 0; ii < rotationalJoints; ii++)
        {
            sensor.AddObservation(jointPositions[ii]); // Add each joint position to observations
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Convert ContinuousActions to a List<float>
        var continuousActions = actionBuffers.ContinuousActions;
        List<float> actionList = new List<float>(continuousActions.Length);
        for (int ii = 0; ii < continuousActions.Length; ii++)
        {
            actionList.Add(continuousActions[ii]);
        }

        // Set the state of the cube with the first 7 actions in the action list
        cube.position = new Vector3(actionList[0], actionList[1], actionList[2]);

        // create a new vector of length 4 and populate it with the elements of indices 3, 4, 5, 6 in actionList
        quat = new Vector4(actionList[4], actionList[5], actionList[6], actionList[3]);  // assume (w, x, y, z) inputs
        quat.Normalize();
        cube.rotation = new Quaternion(quat[0], quat[1], quat[2], quat[3]);
        hand.SetJointPositions(actionList.GetRange(7, rotationalJoints)); // Set joint positions
        SetReward(1f); // Arbitrary reward for each step
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        for (int ii = 0; ii < continuousActions.Length; ii++)
        {
            continuousActions[ii] = Random.Range(-0.1f, 0.1f);
        }
    }
}
