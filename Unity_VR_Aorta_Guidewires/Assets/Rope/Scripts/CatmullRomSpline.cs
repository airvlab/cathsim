using UnityEngine;

namespace Rope
{
    public class CatmullRomSpline
    {
        readonly float[] values;

        public CatmullRomSpline(float[] values)
        {
            this.values = values;
        }

        public float GetValue(float position)
        {
            position = Mathf.Clamp(position, 0, values.Length - 1);

            var i = (int) Mathf.Floor(position);

            if (i == 0)
            {
                return Interpolate(values[0], values[0], values[1], values[2], position - i);
            }

            if (i >= values.Length - 2)
            {
                i = values.Length - 2;
                return Interpolate(values[i - 1], values[i], values[i + 1], values[i + 1], position - i);
            }

            return Interpolate(values[i - 1], values[i], values[i + 1], values[i + 2], position - i);
        }

        static float Interpolate(float p0, float p1, float p2, float p3, float t)
        {
            var v0 = (p2 - p0) / 2f;
            var v1 = (p3 - p1) / 2f;
            var t2 = t * t;
            var t3 = t2 * t;

            return (2 * p1 - 2 * p2 + v0 + v1) * t3 + (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 + v0 * t + p1;
        }

        public static void Test()
        {
            float[] values = {0f, 1f, 2f, 3f, 4f};

            var interpolation = new CatmullRomSpline(values);
            Debug.Log(interpolation.GetValue(-1f));
            Debug.Log(interpolation.GetValue(3.5f));
            Debug.Log(interpolation.GetValue(4f));
            Debug.Log(interpolation.GetValue(5f));
        }
    }
}
