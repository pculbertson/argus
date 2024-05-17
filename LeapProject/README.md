# Unity scene + ml-agents bindings

This project is a simple example of how to use the ml-agents bindings in a Unity scene. The scene is a simple hand where we can read and write to the joint angles using the `leap_demo.py` script.

To run this, in Unity open the scene `Assets/Scenes/LeapDemo.unity`. Build the scene (File -> Build Settings -> Build) and run the executable. Then run the `leap_demo.py` script (with the path to the executable as an argument).

Some basic requirementns on the Python side are listed in `requirements.txt`. 

When running the `leap_demo.py` script, you should see the built Unity app open, and the hand should start moving. The script will print the joint angles of the hand to console. 