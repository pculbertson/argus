using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AgentCallback : Agent
{
    private ArticulationBody hand;
    private int rotationalJoints = 16;

    // Start is called before the first frame update
    void Start()
    {
        hand = this.GetComponentInChildren<ArticulationBody>();
    }

    public override void OnEpisodeBegin()
    {
        // do nothing
    }

    public override void CollectObservations(VectorSensor sensor)
    {
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

        // Set joint angles of the hand per the action buffer
        hand.SetJointPositions(actionList);
        SetReward(1f); // Arbitrary reward for each step
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        for (int ii = 0; ii < continuousActions.Length; ii++)
        {
            continuousActions[ii] = Random.Range(-1.0f, 1.0f);
        }
    }
}
