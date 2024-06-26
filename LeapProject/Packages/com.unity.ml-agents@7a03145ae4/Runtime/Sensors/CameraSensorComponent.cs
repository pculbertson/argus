using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A SensorComponent that creates a <see cref="CameraSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Camera Sensor", (int)MenuGroup.Sensors)]
    public class CameraSensorComponent : SensorComponent, IDisposable
    {
        [HideInInspector, SerializeField, FormerlySerializedAs("camera")]
        Camera m_Camera;

        CameraSensor m_Sensor;

    	public Shader uberReplacementShader;

        /// <summary>
        /// Camera object that provides the data to the sensor.
        /// </summary>
        public Camera Camera
        {
            get { return m_Camera; }
            set { m_Camera = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("sensorName")]
        string m_SensorName = "CameraSensor";

        /// <summary>
        /// Name of the generated <see cref="CameraSensor"/> object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("width")]
        int m_Width = 84;

        /// <summary>
        /// Width of the generated observation.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int Width
        {
            get { return m_Width; }
            set { m_Width = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("height")]
        int m_Height = 84;

        /// <summary>
        /// Height of the generated observation.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int Height
        {
            get { return m_Height; }
            set { m_Height = value; }
        }

        [HideInInspector, SerializeField, FormerlySerializedAs("grayscale")]
        bool m_Grayscale;

        /// <summary>
        /// Whether to generate grayscale images or color.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool Grayscale
        {
            get { return m_Grayscale; }
            set { m_Grayscale = value; }
        }

        [HideInInspector, SerializeField]
        ObservationType m_ObservationType;

        /// <summary>
        /// The type of the observation.
        /// </summary>
        public ObservationType ObservationType
        {
            get { return m_ObservationType; }
            set { m_ObservationType = value; UpdateSensor(); }
        }


        [HideInInspector, SerializeField]
        bool m_RuntimeCameraEnable;


        /// <summary>
        /// Controls the whether the camera sensor's attached camera
        /// is enabled during runtime. Overrides the camera object enabled status.
        /// Disabled for improved performance. Disabled by default.
        /// </summary>
        public bool RuntimeCameraEnable
        {
            get { return m_RuntimeCameraEnable; }
            set { m_RuntimeCameraEnable = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField]
        [Range(1, 50)]
        [Tooltip("Number of camera frames that will be stacked before being fed to the neural network.")]
        int m_ObservationStacks = 1;

        [HideInInspector, SerializeField, FormerlySerializedAs("compression")]
        SensorCompressionType m_Compression = SensorCompressionType.PNG;

        /// <summary>
        /// The compression type to use for the sensor.
        /// </summary>
        public SensorCompressionType CompressionType
        {
            get { return m_Compression; }
            set { m_Compression = value; UpdateSensor(); }
        }

        /// <summary>
        /// Whether to stack previous observations. Using 1 means no previous observations.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int ObservationStacks
        {
            get { return m_ObservationStacks; }
            set { m_ObservationStacks = value; }
        }

        void Start()
        {
            
            if (!uberReplacementShader)
                uberReplacementShader = Shader.Find("Hidden/UberReplacement");

            UpdateSensor();
            OnSceneChange();
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

        /// <summary>
        /// Creates the <see cref="CameraSensor"/>
        /// </summary>
        /// <returns>The created <see cref="CameraSensor"/> object for this component.</returns>
        public override ISensor[] CreateSensors()
        {
            Dispose();        

            m_Sensor = new CameraSensor(m_Camera, m_Width, m_Height, Grayscale, m_SensorName, m_Compression, m_ObservationType);
            if (ObservationStacks != 1)
            {
                return new ISensor[] { new StackingSensor(m_Sensor, ObservationStacks) };
            }
            return new ISensor[] { m_Sensor };
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            int depthMode = 2;
            if (m_Sensor != null)
            {
                m_Sensor.Camera = m_Camera;
                m_Sensor.CompressionType = m_Compression;
                m_Sensor.Camera.enabled = m_RuntimeCameraEnable;
            }

            if (m_Sensor.Camera.name == "depthCam1" || m_Sensor.Camera.name == "depthCam2")
            {
                SetupCameraWithReplacementShader(m_Sensor.Camera, uberReplacementShader, depthMode, Color.white);
            }            
        }

        public void OnSceneChange()
        {
            var renderers = UnityEngine.Object.FindObjectsOfType<Renderer>();
            var mpb = new MaterialPropertyBlock();
            foreach (var r in renderers)
            {
                var id = r.gameObject.GetInstanceID();
                var layer = r.gameObject.layer;
                var tag = r.gameObject.tag;
                r.SetPropertyBlock(mpb);
            }
        }        

        /// <summary>
        /// Clean up the sensor created by CreateSensors().
        /// </summary>
        public void Dispose()
        {
            if (!ReferenceEquals(m_Sensor, null))
            {
                m_Sensor.Dispose();
                m_Sensor = null;
            }
        }
    }
}
