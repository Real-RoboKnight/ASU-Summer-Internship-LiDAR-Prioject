using UnityEngine;
using System.IO;

public class IOScript : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        print("start");
    }

    // Update is called once per frame
    void Update()
    {
        print("loop");
        // TODO: replace hard coded file path
        using (
            StreamWriter writer = File.AppendText(
                "/home/robocat/Documents/Code/ASU/Summer Internship/Unity Project/file.txt"
            )
        )
        {
            writer.WriteLine("loop frame number: " + Time.frameCount);
        }
    }
}
