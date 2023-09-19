using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HapticFeedbackManager : MonoBehaviour
{
    // Duration and intensity for haptic feedback (can vary based on your needs)
    [SerializeField] float hapticDuration = 0.2f;
    [SerializeField] int hapticIntensity = 75;

    // Trigger haptic feedback on a specific event
    public void TriggerHapticFeedback()
    {
        // Check if the platform supports vibration
        if (SystemInfo.supportsVibration)
        {
            // Trigger haptic feedback with specified duration and intensity
            Handheld.Vibrate();
            // For more control, you can use:
            // Handheld.Vibrate(VibrationEffect effect);
        }
        else if (UnityEngine.XR.XRSettings.enabled) // Check if VR is enabled
        {
            // Check if the VR device supports haptic feedback for the left hand controller
            if (UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.LeftHand).TryGetHapticCapabilities(out UnityEngine.XR.HapticCapabilities leftCapabilities))
            {
                if (leftCapabilities.supportsImpulse)
                {
                    // Trigger haptic feedback on the left hand controller
                    UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.LeftHand).SendHapticImpulse(0, hapticIntensity, hapticDuration);
                }
            }

            // Check if the VR device supports haptic feedback for the right hand controller
            if (UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand).TryGetHapticCapabilities(out UnityEngine.XR.HapticCapabilities rightCapabilities))
            {
                if (rightCapabilities.supportsImpulse)
                {
                    // Trigger haptic feedback on the right hand controller
                    UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand).SendHapticImpulse(0, hapticIntensity, hapticDuration);
                }
            }
        }
        else
        {
            Debug.LogWarning("Haptic feedback is not supported on this platform.");
        }
    }
}