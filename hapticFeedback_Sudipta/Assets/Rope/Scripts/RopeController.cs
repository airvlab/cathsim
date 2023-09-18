

using UnityEngine;

namespace Rope
{
    public class RopeController : MonoBehaviour
    {
        // This manages a dynamic rope made of interconnected fragments
        [SerializeField]
        GameObject fragmentPrefab; // Prefab for rope fragments

        [SerializeField]
        int fragmentCount = 80; // The number of rope fragments

        [SerializeField]
        Vector3 interval = new Vector3(0f, 0f, 0.25f); // Spacing between fragments

        [SerializeField]
        float raycastDistance = 1.0f; // // Maximum raycasting distance for collisions. Adjust this value based on your scene.

        [SerializeField]
        private HapticFeedbackManager hapticManager; // Reference to the HapticFeedbackManager script
        
        GameObject[] fragments; // Array to store rope fragments

        float activeFragmentCount; // Number of active (non-kinematic) fragments

        // Arrays to store positions for spline interpolation
        float[] xPositions;
        float[] yPositions;
        float[] zPositions;

        // Catmull-Rom splines for interpolation
        CatmullRomSpline splineX;
        CatmullRomSpline splineY;
        CatmullRomSpline splineZ;

        int splineFactor = 4; // Factor used for spline interpolation

        LayerMask environmentLayerMask; // // LayerMask to specify the environment layer. Declare the LayerMask variable.


        void Start()
        {
            //Initializes the rope by instantiating fragments, setting up connections, and configuring the LineRenderer
            activeFragmentCount = fragmentCount;

            fragments = new GameObject[fragmentCount];

            var position = Vector3.zero;

            for (var i = 0; i < fragmentCount; i++)
            {
                fragments[i] = Instantiate(fragmentPrefab, position, Quaternion.identity);
                fragments[i].tag = "RopeFragment"; // Assign the "RopeFragment" tag to the instantiated fragment
                fragments[i].transform.SetParent(transform);

                var joint = fragments[i].GetComponent<SpringJoint>();
                if (i > 0)
                {
                    joint.connectedBody = fragments[i - 1].GetComponent<Rigidbody>();
                }

                position += interval;
            }

            var lineRenderer = GetComponent<LineRenderer>();
            lineRenderer.positionCount = (fragmentCount - 1) * splineFactor + 1;

            xPositions = new float[fragmentCount];
            yPositions = new float[fragmentCount];
            zPositions = new float[fragmentCount];

            splineX = new CatmullRomSpline(xPositions);
            splineY = new CatmullRomSpline(yPositions);
            splineZ = new CatmullRomSpline(zPositions);

            // Set up the layer mask to target only the specified layer
            environmentLayerMask = LayerMask.GetMask("Environment"); // "Environment" is the name of the layer
        }

        void Update()
        // Updates the rope's behavior, including adjusting its length and handling collisions
        {
            // Adjust the length of the rope based on input
            var vy = Input.GetAxisRaw("Vertical") * 20f * Time.deltaTime;
            activeFragmentCount = Mathf.Clamp(activeFragmentCount + vy, 0, fragmentCount);

            // Iterate through fragments to manage their kinematic state and check for collisions
            for (var i = 0; i < fragmentCount; i++)
            {
                if (i <= fragmentCount - activeFragmentCount)
                {
                    // Deactivate kinematic fragments at the start of the rope
                    fragments[i].GetComponent<Rigidbody>().position = Vector3.zero;
                    fragments[i].GetComponent<Rigidbody>().isKinematic = true;
                }
                else
                {
                    // Activate non-kinematic fragments
                    fragments[i].GetComponent<Rigidbody>().isKinematic = false;
                    // Check for collision with objects in the environment
                    if (CheckCollisionWithEnvironment(fragments[i].transform.position))
                    {
                        // Trigger haptic feedback
                        TriggerHapticFeedback();
                    }
                }
            }
        }

        //Checks for collisions between a fragment and objects in the environment
        bool CheckCollisionWithEnvironment(Vector3 position)
        {
            RaycastHit hit;
            if (Physics.Raycast(position, Vector3.forward, out hit, raycastDistance, environmentLayerMask))
            {
                if (hit.collider.CompareTag("Environment"))
                {
                    return true;
                }
            }
            return false;
        }

        // Triggers haptic feedback based on the platform (VR or non-VR)
        void TriggerHapticFeedback()
        {
            /// Check if the platform supports haptic feedback (VR devices)
            if (UnityEngine.XR.XRSettings.enabled) // Check if VR is enabled
            {
                // Check if the VR device supports haptic feedback for the right hand controller
                if (UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand).TryGetHapticCapabilities(out UnityEngine.XR.HapticCapabilities capabilities))
                {
                    if (capabilities.supportsImpulse)
                    {
                        // Define haptic intensity and duration values
                        float hapticIntensity = 0.5f; // Adjust as needed
                        float hapticDuration = 0.1f; // Adjust as needed

                        // Trigger haptic feedback using impulses
                        UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand).SendHapticImpulse(0, hapticIntensity, hapticDuration);
                    }
                }
            }
            else // For non-VR devices
            {
                // Trigger haptic feedback using vibration
                Handheld.Vibrate();
            }
        }

        void OnCollisionEnter(Collision collision)
        {
            // Check if the collision involves rope fragments
            if (collision.gameObject.CompareTag("RopeFragment"))
            {
                // Trigger haptic feedback
                hapticManager.TriggerHapticFeedback();
            }
        }


        // LateUpdate is called once per frame, after the regular Update method
        // It is used to perform actions that require the results of physics calculations or other update-related operations to be finalized
        void LateUpdate()
        {
            // Copy the positions of the rope fragments' rigidbodies to the LineRenderer
            var lineRenderer = GetComponent<LineRenderer>();

            // No interpolation
            //for (var i = 0; i < fragmentNum; i++)
            //{
            //    renderer.SetPosition(i, fragments[i].transform.position);
            //}

            // Iterate through the rope fragments to gather their current positions
            for (var i = 0; i < fragmentCount; i++)
            {
                var position = fragments[i].transform.position;
                // Store the X, Y, and Z positions in separate arrays for spline interpolation
                xPositions[i] = position.x;
                yPositions[i] = position.y;
                zPositions[i] = position.z;
            }

            // Update the LineRenderer to visualize the rope with interpolated positions
            for (var i = 0; i < (fragmentCount - 1) * splineFactor + 1; i++)
            {
                // Interpolate the positions along the rope using Catmull-Rom splines
                lineRenderer.SetPosition(i, new Vector3(
                    splineX.GetValue(i / (float) splineFactor),
                    splineY.GetValue(i / (float) splineFactor),
                    splineZ.GetValue(i / (float) splineFactor)));
            }
        }
    }
}
