using UnityEngine;

namespace Rope
{
    public class CatmullRomSpline
    {
        readonly float[] values;

        // Constructor: Initializes the spline with an array of values
        public CatmullRomSpline(float[] values)
        {
            this.values = values;
        }

        // GetValue method: Interpolates a value along the Catmull-Rom spline at a given position
        public float GetValue(float position)
        {
            // Clamp the position to ensure it's within the valid range
            position = Mathf.Clamp(position, 0, values.Length - 1);
            // Calculate the segment index and interpolation factor
            int segmentIndex = Mathf.FloorToInt(position);
            float t = position - segmentIndex;

            // Determine the indices of the four control points for interpolation
            int p0 = Mathf.Clamp(segmentIndex - 1, 0, values.Length - 1);
            int p1 = Mathf.Clamp(segmentIndex, 0, values.Length - 1);
            int p2 = Mathf.Clamp(segmentIndex + 1, 0, values.Length - 1);
            int p3 = Mathf.Clamp(segmentIndex + 2, 0, values.Length - 1);

            // Perform Catmull-Rom interpolation using the control points and t parameter
            return Interpolate(values[p0], values[p1], values[p2], values[p3], t);
        }

        // Interpolate method: Performs Catmull-Rom interpolation between four control points
        static float Interpolate(float p0, float p1, float p2, float p3, float t)
        {
            // Calculate interpolation coefficients
            float t2 = t * t;
            float t3 = t2 * t;

            float v0 = (p2 - p0) / 2f;
            float v1 = (p3 - p1) / 2f;

            // Apply the Catmull-Rom spline interpolation formula
            return (2 * p1 - 2 * p2 + v0 + v1) * t3 + (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 + v0 * t + p1;
        }

        // Static method for testing the Catmull-Rom spline
        public static void Test()
        {
            float[] values = { 0f, 1f, 2f, 3f, 4f };

            var interpolation = new CatmullRomSpline(values);
            Debug.Log(interpolation.GetValue(-1f));
            Debug.Log(interpolation.GetValue(3.5f));
            Debug.Log(interpolation.GetValue(4f));
            Debug.Log(interpolation.GetValue(5f));
        }
    }
}
