using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.IO;

// @TODO:
// . support custom color wheels in optical flow via lookup textures
// . support custom depth encoding
// . support multiple overlay cameras
// . tests
// . better example scene(s)

// @KNOWN ISSUES
// . Motion Vectors can produce incorrect results in Unity 5.5.f3 when
//      1) during the first rendering frame
//      2) rendering several cameras with different aspect ratios - vectors do stretch to the sides of the screen

[RequireComponent (typeof(Camera))]
public class ImageSynthesis : MonoBehaviour {

	// pass configuration
	private CapturePass[] capturePasses = new CapturePass[] {
		new CapturePass() { name = "_img" },
		new CapturePass() { name = "_cam1" },
		new CapturePass() { name = "_cam2" },
		// new CapturePass() { name = "_cam3" },		
		new CapturePass() { name = "_depthCam1" },
		new CapturePass() { name = "_depthCam2" },
	};

	struct CapturePass {
		// configuration

		public string name;
		public bool supportsAntialiasing;
		public bool needsRescale;
		public CapturePass(string name_) { name = name_; supportsAntialiasing = true; needsRescale = false; camera = null; }

		// impl
		public Camera camera;
	};
	
	public Shader uberReplacementShader;

	void Start()
	{
		// default fallbacks, if shaders are unspecified
		if (!uberReplacementShader)
			uberReplacementShader = Shader.Find("Hidden/UberReplacement");

		// use real camera to capture final image
		capturePasses[0].camera = GetComponent<Camera>(); // main camera
        foreach (Camera cam in Camera.allCameras)
        {
            switch (cam.name)
            {
                case "cam1":
                    capturePasses[1].camera = cam;
                    break;
                case "cam2":
                    capturePasses[2].camera = cam;
                    break;
                case "depthCam1":
                    capturePasses[3].camera = cam;
                    break;
                case "depthCam2":
                    capturePasses[4].camera = cam;
                    break;
				case "Main Camera": // GetComponent is deterministic for main camera across all machines
					break;
                default:
                    Debug.LogWarning("Camera with unexpected name found: " + cam.name);
                    break;
            }
        }			

		OnCameraChange();
		OnSceneChange();
	}

	void LateUpdate()
	{
		#if UNITY_EDITOR
		if (DetectPotentialSceneChangeInEditor())
			OnSceneChange();
		#endif // UNITY_EDITOR

		// @TODO: detect if camera properties actually changed
		OnCameraChange();
	}

	private Camera CreateDepthCamera(string name)
	{
		var go = new GameObject (name, typeof (Camera));
		go.transform.parent = transform;

		var newCamera = go.GetComponent<Camera>();
		return newCamera;
	}

	static private void SetupCameraWithReplacementShader(Camera cam, Shader shader, int depthMode)
	{
		SetupCameraWithReplacementShader(cam, shader, depthMode, Color.black);
	}

	static private void SetupCameraWithReplacementShader(Camera cam, Shader shader, int depthMode, Color clearColor)
	{
		var cb = new CommandBuffer();
		cb.SetGlobalFloat("_OutputMode", depthMode);
		cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
		cam.AddCommandBuffer(CameraEvent.BeforeFinalPass, cb);
		cam.SetReplacementShader(shader, "");
		cam.backgroundColor = clearColor;
		cam.clearFlags = CameraClearFlags.SolidColor;
	}

	public void OnCameraChange()
	{
		int targetDisplay = 0;
		int depthMode = 2;
		// var cam3 = Camera.allCameras[3];

		// capturePasses[4].camera.RemoveAllCommandBuffers();
		// capturePasses[4].camera.CopyFrom(cam3);

		foreach (var pass in capturePasses) {
			pass.camera.targetDisplay = targetDisplay;
			targetDisplay++;
			Debug.Log($"Target Display: {pass.camera} {targetDisplay}");
		}

		// setup command buffers and replacement shaders
		SetupCameraWithReplacementShader(capturePasses[3].camera, uberReplacementShader, depthMode, Color.white);
		SetupCameraWithReplacementShader(capturePasses[4].camera, uberReplacementShader, depthMode, Color.white);
	}


	public void OnSceneChange()
	{
		var renderers = Object.FindObjectsOfType<Renderer>();
		var mpb = new MaterialPropertyBlock();
		foreach (var r in renderers)
		{
			var id = r.gameObject.GetInstanceID();
			var layer = r.gameObject.layer;
			var tag = r.gameObject.tag;

			mpb.SetColor("_ObjectColor", ColorEncoding.EncodeIDAsColor(id));
			mpb.SetColor("_CategoryColor", ColorEncoding.EncodeLayerAsColor(layer));
			r.SetPropertyBlock(mpb);
		}
	}

	#if UNITY_EDITOR
	private GameObject lastSelectedGO;
	private int lastSelectedGOLayer = -1;
	private string lastSelectedGOTag = "unknown";
	private bool DetectPotentialSceneChangeInEditor()
	{
		bool change = false;
		// there is no callback in Unity Editor to automatically detect changes in scene objects
		// as a workaround lets track selected objects and check, if properties that are 
		// interesting for us (layer or tag) did not change since the last frame
		if (UnityEditor.Selection.transforms.Length > 1)
		{
			// multiple objects are selected, all bets are off!
			// we have to assume these objects are being edited
			change = true;
			lastSelectedGO = null;
		}
		else if (UnityEditor.Selection.activeGameObject)
		{
			var go = UnityEditor.Selection.activeGameObject;
			// check if layer or tag of a selected object have changed since the last frame
			var potentialChangeHappened = lastSelectedGOLayer != go.layer || lastSelectedGOTag != go.tag;
			if (go == lastSelectedGO && potentialChangeHappened)
				change = true;

			lastSelectedGO = go;
			lastSelectedGOLayer = go.layer;
			lastSelectedGOTag = go.tag;
		}

		return change;
	}
	#endif // UNITY_EDITOR
}
