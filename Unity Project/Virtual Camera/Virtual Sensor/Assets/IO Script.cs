using System;
using System.IO;
using UnityEngine;

public class IOScript : MonoBehaviour
{
    const double ANGULAR_RESOLUTION = Math.PI / 1024;

    // TODO: replace hard coded file path
    Camera unityCamera;
    StreamWriter writer;
    double theta = 0;
    double phi = Math.PI;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        print("start");
        writer = File.CreateText("outputFile.csv");
        writer.WriteLine("Frame Number,theta,phi,distance");
        unityCamera = Camera.main;
    }

    // Takes in an angle around the camera and the elevation of the ray and calculates the direction vector
    Vector3 CalculateAngle(double theta, double phi)
    {
        // The unity x vector is right, y is up, and z is forward. This makes a left handed coordinate system
        // theta in [0, 2pi] and phi in [0. pi]
        // phi and theta 0 is the +x vector. Increasing theta rotates counterclockwise on the xz plane.
        // Increasing phi rotates clockwise down from towards +y
        // (cos(θ), sin(θ), sin(φ)).normalize

        return new Vector3(
            (float)(Math.Cos(theta) * Math.Sin(phi)),
            (float)Math.Cos(phi),
            (float)(Math.Sin(theta) * Math.Sin(phi))
        ).normalized;
    }

    string FormatCSV(double theta, double phi, float distance)
    {
        // Frame number, theta, phi, distance (set to 0 for inf)
        return Time.frameCount + "," + theta + "," + phi + "," + distance;
    }

    // Update is called once per frame. Each frame we should do another point
    void Update()
    {
        // Each frame do a vertical scan at a value of theta, starting from the positive y axis down to the negative y axis.
        for (phi = 0; phi < Math.PI; phi += ANGULAR_RESOLUTION)
        {
            Physics.Raycast(
                new Ray(unityCamera.transform.position, CalculateAngle(theta, phi)),
                out RaycastHit raycastHit
            );

            string output = FormatCSV(theta, phi, raycastHit.distance);

            print(output);
            writer.WriteLine(output);
        }

        theta += ANGULAR_RESOLUTION;

        // Quit at the end. Possible to make this call the python analysis script
        if (theta > 2 * Math.PI)
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.ExitPlaymode();
#else
            Application.Quit();
#endif
        }
    }

    void OnDestroy()
    {
        if (writer != null)
        {
            writer.Flush();
        }
    }
}
