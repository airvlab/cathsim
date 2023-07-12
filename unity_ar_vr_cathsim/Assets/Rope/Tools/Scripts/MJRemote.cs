//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2018 Roboti LLC  //
//---------------------------------//

using System;
using System.IO;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;


public class MJRemote : MonoBehaviour 
{

    // socket commands from client
    enum Command : int
    {
        None            = 0,    // null command
        GetInput        = 1,    // send: key, select, active, refpos[3], refquat[4] (40 bytes)
        GetImage        = 2,    // send: rgb image (3*width*height bytes)
        SaveSnapshot    = 3,    // (no data exchange)
        SaveVideoframe  = 4,    // (no data exchange)
        SetCamera       = 5,    // receive: camera index (4 bytes)
        SetQpos         = 6,    // receive: qpos (4*nqpos bytes)
        SetMocap        = 7     // receive: mocap_pos, mocap_quat (28*nmocap bytes)
    }


    // prevent repeated instances
    private static MJRemote instance;
    private MJRemote() {}
    public static MJRemote Instance
    {
        get 
        {
            if( instance==null )
            {
                instance = new MJRemote();
                return instance;
            }
            else
                throw new System.Exception("MJRemote can only be instantiated once");
        }
    }


    // script options
    public string modelFile = "";
    public string tcpAddress = "127.0.0.1";
    public int tcpPort = 1050;

    // GUI
    static GUIStyle style = null;

    // offscreen rendering
    RenderTexture offrt;
    Texture2D offtex;
    int offwidth = 1280;
    int offheight = 720;
    static int snapshots = 0;
    FileStream videofile = null;

    // data from plugin
    int nqpos = 0;
    int nmocap = 0;
    int ncamera = 0;
    int nobject = 0;
    GameObject[] objects;
    Color selcolor;
    GameObject root = null;
    Camera thecamera = null;
    float[] camfov;

    // remote data
    TcpListener listener = null;
    TcpClient client = null;
    NetworkStream stream = null;
    byte[] buffer;
    int buffersize = 0;
    int camindex = -1;
    float lastcheck = 0;

    // input state
    float lastx = 0;        // updated each frame
    float lasty = 0;        // updated each frame
    float lasttime = 0;     // updated on click
    int lastbutton = 0;     // updated on click
    int lastkey = 0;        // cleared on send


    // convert transform from plugin to GameObject
    static unsafe void SetTransform(GameObject obj, MJP.TTransform transform)
    {
        Quaternion q = new Quaternion(0, 0, 0, 1);
        q.SetLookRotation(
            new Vector3(transform.yaxis[0], -transform.yaxis[2], transform.yaxis[1]),
            new Vector3(-transform.zaxis[0], transform.zaxis[2], -transform.zaxis[1])
        );

        obj.transform.localPosition = new Vector3(-transform.position[0], transform.position[2], -transform.position[1]);
        obj.transform.localRotation = q;
        obj.transform.localScale = new Vector3(transform.scale[0], transform.scale[2], transform.scale[1]);
    }


    // convert transform from plugin to Camera
    static unsafe void SetCamera(Camera cam, MJP.TTransform transform)
    {
        Quaternion q = new Quaternion(0, 0, 0, 1);
        q.SetLookRotation(
            new Vector3(transform.zaxis[0], -transform.zaxis[2], transform.zaxis[1]),
            new Vector3(-transform.yaxis[0], transform.yaxis[2], -transform.yaxis[1])
        );

        cam.transform.localPosition = new Vector3(-transform.position[0], transform.position[2], -transform.position[1]);
        cam.transform.localRotation = q;
    }

    
    // GUI
    private void OnGUI()
    {
        // set style once
        if( style==null )
        {
            style = GUI.skin.textField;
            style.normal.textColor = Color.white;

            // scale font size with DPI
            if( Screen.dpi<100 )
                style.fontSize = 14;
            else if( Screen.dpi>300 )
                style.fontSize = 34;
            else
                style.fontSize = Mathf.RoundToInt(14 + (Screen.dpi-100.0f)*0.1f);
        }

        // show connected status
        if( client!=null && client.Connected )
            GUILayout.Label("Connected", style);
        else
            GUILayout.Label("Waiting", style);

        // save lastkey
        if( Event.current.isKey )
            lastkey = (int)Event.current.keyCode;
    }


    // initialize
    unsafe void Start()
    {
        // set selection color
        selcolor = new Color(0.5f, 0.5f, 0.5f, 1);

        // initialize plugin
        MJP.Initialize();
        MJP.LoadModel(Application.streamingAssetsPath + "/" + modelFile);

        // get number of renderable objects, allocate map
        MJP.TSize size;
        MJP.GetSize(&size);
        nqpos = size.nqpos;
        nmocap = size.nmocap;
        ncamera = size.ncamera;
        nobject = size.nobject;
        objects = new GameObject[nobject];

        // get root
        root = GameObject.Find("MuJoCo");
        if( root==null )
            throw new System.Exception("MuJoCo root object not found");

        // get camera under root
        int nchild = root.transform.childCount;
        for( int i=0; i<nchild; i++ )
        {
            thecamera = root.transform.GetChild(i).gameObject.GetComponent<Camera>();
            if( thecamera!=null )
                break;
        }
        if( thecamera==null )
            throw new System.Exception("No camera found under MuJoCo root object");

        // make map of renderable objects
        for( int i=0; i<nobject; i++ )
        {
            // get object name
            StringBuilder name = new StringBuilder(100);
            MJP.GetObjectName(i, name, 100);

            // find corresponding GameObject
            for( int j=0; j<nchild; j++ )
                if( root.transform.GetChild(j).name == name.ToString() )
                {
                    objects[i] = root.transform.GetChild(j).gameObject;
                    break;
                }

            // set initial state
            if( objects[i] )
            {
                MJP.TTransform transform;
                int visible;
                int selected;
                MJP.GetObjectState(i, &transform, &visible, &selected);
                SetTransform(objects[i], transform);
                objects[i].SetActive( visible > 0 );
            }
        }

        // get camera fov and offscreen resolution
        camfov = new float[ncamera+1];
        for( int i=-1; i<ncamera; i++ )
        {
            MJP.TCamera cam;
            MJP.GetCamera(i, &cam);
            camfov[i+1] = cam.fov;

            // plugin returns offscreen width and height for all cameras
            offwidth = cam.width;
            offheight = cam.height;
        }

        // prepare offscreen rendering
        offtex = new Texture2D(offwidth, offheight, TextureFormat.RGB24, false);
        offrt = new RenderTexture(offwidth, offheight, 24);
        offrt.width = offwidth;
        offrt.height = offheight;
        offrt.Create();

        // synchronize time
        MJP.SetTime(Time.time);

        // preallocate buffer with maximum possible message size
        buffersize = Math.Max(4, Math.Max(4*nqpos, 28*nmocap));
        buffer = new byte[buffersize];

        // start listening for connections
        listener = new TcpListener(System.Net.IPAddress.Parse(tcpAddress), tcpPort);
        listener.Start();
    }
    

    // render to texture
    private void RenderToTexture()
    {
        // set to offscreen and render
        thecamera.targetTexture = offrt;
        thecamera.Render();

        // read pixels in regular texure and save
        RenderTexture.active = offrt;
        offtex.ReadPixels(new Rect(0, 0, offwidth, offheight), 0, 0);
        offtex.Apply();

        // restore state
        RenderTexture.active = null;
        thecamera.targetTexture = null;
    }


    // per-frame mouse input; called from Update
    unsafe void ProcessMouse()
    {
        // get modifiers
        bool alt = Input.GetKey(KeyCode.LeftAlt) || Input.GetKey(KeyCode.RightAlt);
        bool shift = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        bool control = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);

        // get button pressed, swap left-right on alt
        int buttonpressed = 0;
        if( Input.GetMouseButton(0) )           // left
            buttonpressed = (alt ? 2 : 1);
        if( Input.GetMouseButton(1) )           // right
            buttonpressed = (alt ? 1 : 2);
        if( Input.GetMouseButton(2) )           // middle
            buttonpressed = 3;

        // get button click, swap left-right on alt
        int buttonclick = 0;
        if( Input.GetMouseButtonDown(0) )       // left
            buttonclick = (alt ? 2 : 1);
        if( Input.GetMouseButtonDown(1) )       // right
            buttonclick = (alt ? 1 : 2);
        if( Input.GetMouseButtonDown(2) )       // middle
            buttonclick = 3;

        // click
        if( buttonclick>0 )
        {
            // set perturbation state
            int newstate = 0;
            if( control )
            {
                // determine new perturbation state
                if( buttonclick==1 )
                    newstate = 2;              // rotate
                else if( buttonclick==2 )
                    newstate = 1;              // move

                // get old perturbation state
                MJP.TPerturb current;
                MJP.GetPerturb(&current);

                // syncronize if starting perturbation now
                if( newstate>0 && current.active==0 )
                    MJP.PerturbSynchronize();
            }
            MJP.PerturbActive(newstate);

            // process double-click
            if( buttonclick==lastbutton && Time.fixedUnscaledTime-lasttime<0.25 )
            {
                // relative screen position and aspect ratio
                float relx = Input.mousePosition.x / Screen.width;
                float rely = Input.mousePosition.y / Screen.height;
                float aspect = (float)Screen.width / (float)Screen.height;

                // left: select body
                if( buttonclick==1 )
                    MJP.PerturbSelect(relx, rely, aspect);

                // right: set lookat
                else if( buttonclick==2 )
                    MJP.CameraLookAt(relx, rely, aspect);
            }

            // save mouse state
            lastx = Input.mousePosition.x;
            lasty = Input.mousePosition.y;
            lasttime = Time.fixedUnscaledTime;
            lastbutton = buttonclick;
        }

        // left or right drag: manipulate camera or perturb
        if( buttonpressed==1 || buttonpressed==2 )
        {
            // compute relative displacement and modifier
            float reldx = (Input.mousePosition.x - lastx) / Screen.height;
            float reldy = (Input.mousePosition.y - lasty) / Screen.height;
            int modifier = (shift ? 1 : 0);

            // perturb
            if( control )
            {
                if (buttonpressed == 1)
                    MJP.PerturbRotate(reldx, -reldy, modifier);
                else
                    MJP.PerturbMove(reldx, -reldy, modifier);
            }

            // camera
            else
            {
                if( buttonpressed==1 )
                    MJP.CameraRotate(reldx, -reldy);
                else
                    MJP.CameraMove(reldx, -reldy, modifier);
            }
        }

        // middle drag: zoom camera
        if( buttonpressed==3 )
        {
            float reldy = (Input.mousePosition.y - lasty) / Screen.height;
            MJP.CameraZoom(-reldy);
        }

        // scroll: zoom camera
        if( Input.mouseScrollDelta.y!=0 )
            MJP.CameraZoom(-0.05f * Input.mouseScrollDelta.y);

        // save position
        lastx = Input.mousePosition.x;
        lasty = Input.mousePosition.y;

        // release left or right: stop perturb
        if( Input.GetMouseButtonUp(0) || Input.GetMouseButtonUp(1) )
            MJP.PerturbActive(0);
    }


    // update Unity representation of MuJoCo model
    unsafe private void UpdateModel()
    {
        MJP.TTransform transform;

        // update object states
        for( int i=0; i<nobject; i++ )
            if( objects[i] )
            {
                // set transform and visibility
                int visible;
                int selected;
                MJP.GetObjectState(i, &transform, &visible, &selected);
                SetTransform(objects[i], transform);
                objects[i].SetActive(visible>0);

                // set emission color
                if( selected>0 )
                    objects[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", selcolor);
                else
                    objects[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", Color.black);
            }

        // update camera
        MJP.GetCameraState(camindex, &transform);
        SetCamera(thecamera, transform);
        thecamera.fieldOfView = camfov[camindex+1];
    }


    // check if connection is still alive
    private bool CheckConnection()
    {
        try
        {
            if( client!=null && client.Client!=null && client.Client.Connected )
            {
                if( client.Client.Poll(0, SelectMode.SelectRead) )
                {
                    if( client.Client.Receive(buffer, SocketFlags.Peek)==0 )
                        return false;
                    else
                        return true;
                }
                else
                    return true;
            }
            else
                return false;
        }
        catch
        {
            return false;
        }
    }


    // read requested number of bytes from socket
    void ReadAll(int n)
    {
        int i = 0;
        while( i<n )
            i += stream.Read(buffer, i, n-i);
    }


    // per-frame update
    unsafe void Update ()
    {
        // mouse interaction
        ProcessMouse();
        UpdateModel();

        // check conection each 0.1 sec
        if( lastcheck+0.1f<Time.time )
        {
            // broken connection: clear
            if( !CheckConnection() )
            {
                client = null;
                stream = null;
            }

            lastcheck = Time.time; 
        }

        // not connected: accept connection if pending
        if( client==null || !client.Connected )
        {
            if( listener.Pending() )
            {
                // make connection
                client = listener.AcceptTcpClient();
                stream = client.GetStream();

                // send 20 bytes: nqpos, nmocap, ncamera, width, height
                stream.Write(BitConverter.GetBytes(nqpos), 0, 4);
                stream.Write(BitConverter.GetBytes(nmocap), 0, 4);
                stream.Write(BitConverter.GetBytes(ncamera), 0, 4);
                stream.Write(BitConverter.GetBytes(offwidth), 0, 4);
                stream.Write(BitConverter.GetBytes(offheight), 0, 4);
            }
        }

        // data available: handle communication
        while( client!=null && client.Connected && stream!=null && stream.DataAvailable )
        {
            // get command
            ReadAll(4);
            int cmd = BitConverter.ToInt32(buffer, 0);

            // process command
            switch( (Command)cmd )
            {
                // GetInput: send lastkey, select, active, refpos[3], refquat[4]
                case Command.GetInput:
                    MJP.TPerturb perturb;
                    MJP.GetPerturb(&perturb);
                    stream.Write(BitConverter.GetBytes(lastkey), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.select), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.active), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refpos[0]), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refpos[1]), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refpos[2]), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refquat[0]), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refquat[1]), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refquat[2]), 0, 4);
                    stream.Write(BitConverter.GetBytes(perturb.refquat[3]), 0, 4);
                    lastkey = 0;
                    break;

                // GetImage: send 3*width*height bytes
                case Command.GetImage:
                    RenderToTexture();
                    stream.Write(offtex.GetRawTextureData(), 0, 3*offwidth*offheight);
                    break;

                // SaveSnapshot: no data exchange
                case Command.SaveSnapshot:
                    RenderToTexture();
                    byte[] bytes = offtex.EncodeToPNG();
                    File.WriteAllBytes(Application.streamingAssetsPath + "/../../" + "img_" + 
                                       snapshots + ".png", bytes);
                    snapshots++;
                    break;

                // SaveVideoframe: no data exchange
                case Command.SaveVideoframe:
                    if( videofile==null )
                        videofile = new FileStream(Application.streamingAssetsPath + "/../../" + "video.raw",
                                                   FileMode.Create, FileAccess.Write);
                    RenderToTexture();
                    videofile.Write(offtex.GetRawTextureData(), 0, 3*offwidth*offheight);
                    break;

                // SetCamera: receive camera index
                case Command.SetCamera:
                    ReadAll(4);
                    camindex = BitConverter.ToInt32(buffer, 0);
                    camindex = Math.Max(-1, Math.Min(ncamera-1, camindex));
                    break;

                // SetQpos: receive qpos vector
                case Command.SetQpos:
                    if( nqpos>0 )
                    {
                        ReadAll(4*nqpos);
                        fixed( byte* qpos=buffer )
                        {
                            MJP.SetQpos((float*)qpos);
                        }
                        MJP.Kinematics();
                        UpdateModel();
                    }
                    break;

                // SetMocap: receive mocap_pos and mocap_quat vectors
                case Command.SetMocap:
                    if( nmocap>0 )
                    {
                        ReadAll(28*nmocap);
                        fixed( byte* pos=buffer, quat=&buffer[12*nmocap] )
                        {
                            MJP.SetMocap((float*)pos, (float*)quat);
                        }
                        MJP.Kinematics();
                        UpdateModel();
                    }
                    break;
            }
        }
    }


    // cleanup
    void OnApplicationQuit()
    {
        // free plugin
        MJP.Close();

        // close tcp listener
        listener.Stop();

        // close file
        if( videofile!=null )
            videofile.Close();

        // free render texture
        offrt.Release();
    }
}
