# argus
Yet another vision-based pose estimator for the c u b e 

## Unity Sharp Edges
1. Make sure your Unity version is one of the newer ones from 2023 or later.
2. The correct ML-Agents package in Unity is 3.0.0 (the Release 21 version), which you can't find from searching regularly in the package manager. Search by git repo instead, and paste the link
    ```
    git+https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents#release_21
    ```
3. To actually fix a geometry properly, make sure on the `Articulation Body` that the `Immovable` option is checked. This will also remove its degrees of freedoms when reading its state.
4. When adding cameras into the scene, make sure you turn off the `Audio Listener` functionality. For debugging purposes, it's also useful to set different `Display` options for each camera. You can also set the camera background as well as its clipping planes.