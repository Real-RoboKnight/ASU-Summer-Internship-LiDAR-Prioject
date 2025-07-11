using System;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

public class IOScript : MonoBehaviour
{
    public String pythonPath;
    public String pythonFile;
    public String outputPath;
    public const int PERIOD = 30; // capture data for ten frames then call python

    string tmpPath;

    const double ANGULAR_RESOLUTION = Math.PI / 512;
    const double TWO_PI = 2.0 * Math.PI;

    // TODO: replace hard coded file path
    Camera unityCamera;
    StreamWriter writer;
    double[] thetaBounds = new double[PERIOD + 1];

    void Start()
    {
        tmpPath = Path.GetTempFileName();
        // Process.Start(@"/usr/bin/rm", $"-rf \"{outputPath}\"");

        print("start " + tmpPath);
        writer = File.CreateText(tmpPath);
        writer.WriteLine("Frame Number,theta,phi,distance");
        unityCamera = Camera.main;

        // Precompute theta bounds based on the period.
        for (int i = 0; i <= PERIOD; i++)
        {
            thetaBounds[i] = 0 + i * (TWO_PI / PERIOD);
        }
        Directory.Delete(outputPath, true);
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
            print(
                $"Called Python.\n {pythonPath}\t\"{pythonFile}\" --csv-file \"{tmpPath}\" --output-dir \"{outputPath}{Time.frameCount}/\""
            );

            Process.Start(
                pythonPath,
                $"\"{pythonFile}\" --csv-file \"{tmpPath}\" --output-dir \"{outputPath}{Time.frameCount}/\""
            );
            // File.Delete(tmpPath);

            tmpPath = Path.GetTempFileName();
            writer = File.CreateText(tmpPath);
            writer.WriteLine("Frame Number,theta,phi,distance");
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

    public void Move(InputAction.CallbackContext callbackConstant)
    {
        print(callbackConstant);
        Vector2 direction = callbackConstant.ReadValue<Vector2>();
        Camera.main.transform.position =
            Camera.main.transform.position
            + Camera.main.transform.rotation * new Vector3(direction.x, 0, direction.y);
    }

    public void Jump(InputAction.CallbackContext callbackConstant)
    {
        print(callbackConstant);
        Camera.main.transform.position = Camera.main.transform.position + transform.up;
    }

    public void Crouch(InputAction.CallbackContext callbackConstant)
    {
        print(callbackConstant);
        Camera.main.transform.position = Camera.main.transform.position - transform.up;
    }

    public void Look(InputAction.CallbackContext callbackConstant)
    {
        print(callbackConstant);
        Vector2 direction = callbackConstant.ReadValue<Vector2>();
        UnityEngine.Debug.Log(direction);
        Camera.main.transform.Rotate(new Vector3(-(direction.y / 50f), direction.x / 50f, 0));
    }
}
