using UnityEngine;

namespace Rope
{
    [RequireComponent(typeof(Rigidbody))]
    public class MouseDraggable : MonoBehaviour
    {
        private Rigidbody rigid;

        private bool isDragging = false;
        private Vector3 offset;

        private void Awake()
        {
            // Get a reference to the Rigidbody component attached to the GameObject
            rigid = GetComponent<Rigidbody>();
        }

        private void Update()
        {
            if (isDragging)
            {
                // Calculate the mouse position in world coordinates with a fixed z-coordinate
                Vector3 mousePos = Camera.main.ScreenToWorldPoint(
                    new Vector3(Input.mousePosition.x, Input.mousePosition.y, rigid.position.z));
                Vector3 destination = mousePos + offset;

                // Smoothly move the object towards the destination using physics-based velocity
                Vector3 velocity = (destination - rigid.position) * 50f;
                rigid.velocity = velocity;
            }
        }

        private void OnMouseDown()
        {
            // Calculate the offset between the object's position and the mouse click point
            Vector3 screenPoint = Camera.main.WorldToScreenPoint(transform.position);
            offset = transform.position - Camera.main.ScreenToWorldPoint(
                new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenPoint.z));

            isDragging = true;

            // Disable physics simulation while dragging to allow for smooth movement
            rigid.isKinematic = true;
        }

        private void OnMouseUp()
        {
            isDragging = false;

            // Re-enable physics simulation when dragging is done to resume normal behaviour
            rigid.isKinematic = false;
        }
    }
}
