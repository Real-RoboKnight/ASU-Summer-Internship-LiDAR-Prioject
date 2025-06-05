using System;
using System.IO;
using UnityEngine;

public class IOScript : MonoBehaviour
{
    const double ANGULAR_RESOLUTION = Math.PI / 64;

    // TODO: replace hard coded file path
    Camera unityCamera;
    StreamWriter writer;
    double theta = 0;
    double phi = Math.PI;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        print("start");
        writer = File.CreateText("outputFile.txt");
        unityCamera = Camera.main;
    }

    Vector3 CalculateAngle(double theta, double phi)
    {
        // theta in [0, 2pi] and phi in [-pi. pi]
        // phi and theta 0 is the +x vector. Increasing theta rotates counterclockwise on the xz plane. Increasing phi rotates counterclockwise up towards +y
        // (cos(θ), sin(θ), sin(φ)).normalize

        return new Vector3(
            (float)Math.Sin(theta),
            (float)Math.Cos(theta),
            (float)Math.Sin(phi)
        ).normalized;
    }

    // Update is called once per frame. Each frame we should do another point
    void Update()
    {
        for (phi = Math.PI; phi > -Math.PI; phi -= ANGULAR_RESOLUTION)
        {
            if (
                Physics.Raycast(
                    new Ray(unityCamera.transform.position, CalculateAngle(theta, phi)),
                    out RaycastHit raycastHit
                )
            )
            {
                string output =
                    raycastHit.collider.name
                    + " was hit, theta was "
                    + theta
                    + ", rho was "
                    + phi
                    + ", distance was "
                    + raycastHit.distance;
                print(output);
                writer.WriteLine(output);
            }
            else
            {
                string output = "Nothing was hit, theta was " + theta + ", rho was " + phi;
                print(output);
                writer.WriteLine(output);
            }
        }
        theta += ANGULAR_RESOLUTION;
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
            writer.Close();
        }
    }
}
