using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;

public class IOScript : MonoBehaviour
{
    const double ANGULAR_RESOLUTION = Math.PI / 64;
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
        Process.Start(@"/usr/bin/rm", "-rf \"/home/robocat/Documents/Code/ASU/Summer Internship/lidar_analysis_results/\"");
    }

    Vector3 CalculateAngle(double theta, double phi)
    {
        // Validate input angles
        if (phi < 0 || phi > Math.PI)
        {
            UnityEngine.Debug.LogWarning($"Invalid phi value: {phi}. Clamping to [0, π]");
            phi = Mathf.Clamp((float)phi, 0, (float)Math.PI);
        }

        // Normalize theta to [0, 2π]
        theta = theta % TWO_PI;
        if (theta < 0)
            theta += TWO_PI;

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

        return new Vector3(cosTheta * sinPhi, cosPhi, sinTheta * sinPhi).normalized;
    }

    string FormatCSV(double theta, double phi, float distance)
    {
        return Time.frameCount + "," + theta + "," + phi + "," + distance;
    }

    void Update()
    {
        print("frame");

        writer = File.CreateText($"/tmp/Unity/outputFile/{Time.frameCount}.csv");
        writer.WriteLine("Frame Number,theta,phi,distance");
        for (theta = 0; theta < TWO_PI; theta += ANGULAR_RESOLUTION)
        {
            for (phi = 0; phi < Math.PI; phi += ANGULAR_RESOLUTION)
            {
                Vector3 direction = CalculateAngle(theta, phi);
                if (
                    Physics.Raycast(
                        unityCamera.transform.position,
                        direction,
                        out RaycastHit raycastHit
                    )
                )
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
        }
        writer.Flush();
        writer.Close();

        // Only process 6 images per second.
        if (Time.frameCount % 10 == 0)
        {
            print(
                $"\"/home/robocat/Documents/Code/ASU/Summer Internship/.venv/bin/python\" \"/home/robocat/Documents/Code/ASU/Summer Internship/python-tool/advanced_lidar_processing.py\" --csv-file /tmp/Unity/outputFile/{Time.frameCount}.csv --output-dir \"/home/robocat/Documents/Code/ASU/Summer Internship/lidar_analysis_results/{Time.frameCount}/\""
            );

            Process.Start(
                @"/home/robocat/Documents/Code/ASU/Summer Internship/.venv/bin/python",
                $"\"/home/robocat/Documents/Code/ASU/Summer Internship/python-tool/advanced_lidar_processing.py\" --csv-file /tmp/Unity/outputFile/{Time.frameCount}.csv --output-dir \"/home/robocat/Documents/Code/ASU/Summer Internship/lidar_analysis_results/{Time.frameCount}/\""
            );
        }
        // Background process the python
        // .WaitForExit();
    }

    void OnDestroy()
    {
        // if (writer != null)
        // {
        //     writer.Flush();
        //     // we don't want data loss
        //     writer.Close();
        //     writer = null;
        // }
    }
}
