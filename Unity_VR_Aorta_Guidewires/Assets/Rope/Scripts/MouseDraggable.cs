using UnityEngine;

namespace Rope
{
    [RequireComponent(typeof(Rigidbody))]
    public class MouseDraggable : MonoBehaviour
    {
        Vector3 screenPoint;
        Vector3 offset;

        bool dragging;

        Rigidbody rigid;

        void Awake()
        {
            rigid = GetComponent<Rigidbody>();
        }

        void OnMouseDown()
        {
            screenPoint = Camera.main.WorldToScreenPoint(transform.position);
            offset = transform.position - Camera.main.ScreenToWorldPoint(
                new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenPoint.z));

            dragging = true;
        }

        void OnMouseUp()
        {
            dragging = false;
        }

        void FixedUpdate()
        {
            if (dragging)
            {
                var point = new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenPoint.z);
                var dest = Camera.main.ScreenToWorldPoint(point) + offset;

                rigid.AddForce((dest - rigid.position) * 50f);
                rigid.velocity *= 0.8f;
            }
        }
    }
}
