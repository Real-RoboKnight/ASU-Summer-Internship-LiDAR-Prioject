using System;
using System.IO;
using UnityEngine;

public class IOScript : MonoBehaviour
{
    const double ANGULAR_RESOLUTION = Math.PI / 1024;
    const double TWO_PI = 2.0 * Math.PI;

    // TODO: replace hard coded file path
    Camera unityCamera;
    StreamWriter writer;
    double theta = 0;
    double phi = Math.PI;
    Vector3[] directionCache;
    int cacheSize;

    void Start()
    {
        print("start");
        writer = File.CreateText("outputFile.csv");
        writer.WriteLine("Frame Number,theta,phi,distance");
        unityCamera = Camera.main;

        // Pre-calculate direction vectors for better performance
        cacheSize = (int)(Math.PI / ANGULAR_RESOLUTION);
        directionCache = new Vector3[cacheSize];
        for (int i = 0; i < cacheSize; i++)
        {
            double currentPhi = i * ANGULAR_RESOLUTION;
            directionCache[i] = CalculateAngle(0, currentPhi);
        }
    }

    Vector3 CalculateAngle(double theta, double phi)
    {
        // Validate input angles
        if (phi < 0 || phi > Math.PI)
        {
            Debug.LogWarning($"Invalid phi value: {phi}. Clamping to [0, π]");
            phi = Mathf.Clamp((float)phi, 0, (float)Math.PI);
        }

        // Normalize theta to [0, 2π]
        theta = theta % TWO_PI;
        if (theta < 0) theta += TWO_PI;

        // Use cached direction if possible
        if (Math.Abs(theta) < 1e-6) // If theta is basically 0
        {
            int index = (int)(phi / ANGULAR_RESOLUTION);
            if (index >= 0 && index < cacheSize)
            {
                return directionCache[index];
            }
        }

        // Calculate direction vector - pre-calculate trig functions
        float sinPhi = (float)Math.Sin(phi);
        float cosTheta = (float)Math.Cos(theta);
        float sinTheta = (float)Math.Sin(theta);
        float cosPhi = (float)Math.Cos(phi);

        return new Vector3(
            cosTheta * sinPhi,
            cosPhi,
            sinTheta * sinPhi
        ).normalized;
    }

    string FormatCSV(double theta, double phi, float distance)
    {
        return Time.frameCount + "," + theta + "," + phi + "," + distance;
    }

    void Update()
    {
        // Each frame do a vertical scan at a value of theta
        for (phi = 0; phi < Math.PI; phi += ANGULAR_RESOLUTION)
        {
            Vector3 direction = CalculateAngle(theta, phi);
            if (Physics.Raycast(unityCamera.transform.position, direction, out RaycastHit raycastHit))
            {
                string output = FormatCSV(theta, phi, raycastHit.distance);
                writer.WriteLine(output);
            }
            else
            {
                string output = FormatCSV(theta, phi, 0f); // No hit = infinite distance
                writer.WriteLine(output);
            }
        }

        theta += ANGULAR_RESOLUTION;

        if (theta > TWO_PI)
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
            // we don't want data loss
            writer.Close();
            writer = null;
        }
    }
}