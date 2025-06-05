using System;
using System.IO;
using UnityEngine;

public class IOScript : MonoBehaviour
{
    // TODO: replace hard coded file path
    Camera unityCamera;
    StreamWriter writer;
    Ray ray;

    Vector3 angle;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        print("start");
        writer = File.CreateText("outputFile.txt");
        unityCamera = Camera.main;
    }

    // Update is called once per frame
    void Update()
    {
        print("loop");
        writer.WriteLine("loop frame number: " + Time.frameCount);

        angle = new Vector3(0, 0, 1); //Hardcoded forward for now
        if (
            Physics.Raycast(
                new Ray(unityCamera.transform.position, angle),
                out RaycastHit raycastHit
            )
        )
        {
            String output =
                raycastHit.collider.name
                + " was hit, angle was "
                + angle.ToString()
                + ", distance was "
                + raycastHit.distance;
            print(output);
            writer.WriteLine(output);
        }

        writer.Flush();
    }

    void OnDestroy()
    {
        if (writer != null)
        {
            writer.Close();
        }
    }
}
