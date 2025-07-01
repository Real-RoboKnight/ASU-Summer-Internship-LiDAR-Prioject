using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;

public class IOScript : MonoBehaviour
{
    const double ANGULAR_RESOLUTION = Math.PI / 64;
    const double TWO_PI = 2.0 * Math.PI;
    const int PERIOD = 10; // capture data for ten frames then call python

    // TODO: replace hard coded file path
    Camera unityCamera;
    StreamWriter writer;
    double[] thetaBounds = new double[PERIOD + 1];

    void Start()
    {
        print("start");
        writer = File.CreateText("/tmp/Unity/outputFile/0.csv");
        writer.WriteLine("Frame Number,theta,phi,distance");
        unityCamera = Camera.main;

        // Precompute theta bounds based on the period.
        for (int i = 0; i <= PERIOD; i++)
        {
            thetaBounds[i] = 0 + i * (TWO_PI / PERIOD);
        }

        Process.Start(
            @"/usr/bin/rm",
            "-rf \"/home/robocat/Documents/Code/ASU/Summer Internship/lidar_analysis_results/\""
        );
        Process.Start(
            @"/usr/bin/rm",
            "-rf \"/home/robocat/Documents/Code/ASU/Summer Internship/lidar_analysis_results/\""
        );
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

        // After setup, wait till the counter resets.
        if (Time.frameCount < PERIOD)
            return;
        // Only process 1 images per PERIOD frames.
        if (Time.frameCount % PERIOD == 0)
        {
            writer.Flush();
            writer.Close();
            writer = File.CreateText($"/tmp/Unity/outputFile/{Time.frameCount}.csv");
            writer.WriteLine("Frame Number,theta,phi,distance");

            print("Called Python.");
            Process.Start(
                @"/home/robocat/Documents/Code/ASU/Summer Internship/.venv/bin/python",
                $"\"/home/robocat/Documents/Code/ASU/Summer Internship/python-tool/lidar_visualize_new.py\" --csv-file /tmp/Unity/outputFile/{Time.frameCount - PERIOD}.csv --output-dir \"/home/robocat/Documents/Code/ASU/Summer Internship/lidar_analysis_results/{Time.frameCount}/\""
            );
        }

        for (
            double theta = thetaBounds[Time.frameCount % PERIOD];
            theta < thetaBounds[(Time.frameCount % PERIOD) + 1];
            theta += ANGULAR_RESOLUTION
        )
        {
            for (double phi = 0; phi < Math.PI; phi += ANGULAR_RESOLUTION)
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
    }

    void OnDestroy() { }
}
