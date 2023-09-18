//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2018 Roboti LLC  //
//---------------------------------//

#pragma once


// cross-platform export
#if defined(MJP_BUILD)
	#ifdef _WIN32
		#define MJPAPI __declspec(dllexport)
	#else
		#define MJPAPI __attribute__ ((visibility("default")))
	#endif

// cross-platform import
#else
	#ifdef _WIN32
		#define MJPAPI __declspec(dllimport)
	#else
		#define MJPAPI
	#endif

#endif


// this is a C API
#ifdef __cplusplus
extern "C"
{
#endif


//------------------------------ Type definitions ---------------------------------------

typedef enum					// code returned by API functions
{
	RES_OK			= 0,		// success
	RES_MUJOCOERROR,			// MuJoCo error (see mjp_getErrorText)
	RES_EXCEPTION,				// unknown exception
	RES_NOINIT,					// plugin not initialized
	RES_NOMODEL,				// model not present
	RES_BADMODEL,				// failed to load model
	RES_BADINDEX,				// index out of range
	RES_NULLPOINTER,			// null pointer
	RES_SMALLBUFFER,			// buffer size too small
	RES_NOFILENAME,				// file name null or empty
	RES_REPEATNAME				// repeated object name
} mjpResult;


typedef enum					// model element type (subset of mjtObj)
{
    ELEM_CAMERA		= 7,        // camera
    ELEM_LIGHT,					// light
    ELEM_MESH,					// mesh
    ELEM_HFIELD,				// height field
    ELEM_TEXTURE,				// texture
    ELEM_MATERIAL				// material
} mjpElement;


typedef enum  					// renderable object category
{
	CAT_PERTBAND	= 0,		// perturbation band
	CAT_PERTCUBE,				// perturbation cube
	CAT_GEOM,					// geom
	CAT_SITE,					// site
	CAT_TENDON					// tendon
} mjpCategory;


struct _mjpOption				// subset of physics options
{
	int gravity;				// enable gravity
	int equality;				// enable equality constraints
	int limit;					// enable joint and tendon limits
};
typedef struct _mjpOption mjpOption;


struct _mjpSize					// number of model elements by type
{
	// copied from mjModel
	int nbody;					// number of bodies
	int nqpos;					// number of position coordinates
	int nmocap;					// number of mocap bodies
	int nmaterial;				// number of materials
	int ntexture;				// number of textures
	int nlight;					// number of lights (in addition to headlight)
	int ncamera;				// number of cameras (in addition to main camera)
	int nkeyframe;				// number of keyframes

	// computed by plugin
	int nobject;				// number of renderable objects
};
typedef struct _mjpSize mjpSize;


struct _mjpObject				// renderable object descriptor
{
	// common
	int category;				// object caregory (mjpCategory)
	int geomtype;				// geom type (mjtGeom)
	int material;				// material id; -1: none
	int dataid;					// mesh or hfield id; -1: none
	float color[4];				// local color; override material if different from (.5 .5 .5 1)

	// mesh specific
	int mesh_shared;			// index of object containing shared mesh; -1: none
	int mesh_nvertex;			// number of mesh vertices
	int mesh_nface;				// number of mesh triangles
	float* mesh_position;		// vertex position data				(nvertex x 3)
	float* mesh_normal;			// vertex normal data				(nvertex x 3)
	float* mesh_texcoord;		// vertex texture coordinate data	(nvertex x 2)
	int* mesh_face;				// face index data					(nface x 3)

	// height field specific
	int hfield_nrow;			// number of rows (corresponding to x-axis)
	int hfield_ncol;			// number of columns (corresponding to y-axis)
	float* hfield_data;			// elevation data					(ncol x nrow)
};
typedef struct _mjpObject mjpObject;


struct _mjpMaterial				// material descriptor
{
	int texture;				// texture id; -1: none
	int texuniform;				// uniform texture cube mapping
	float texrepeat[2];			// repetition number of 2d textures (x,y)
	float color[4];				// main color
	float emission;				// emission coefficient
	float specular;				// specular coefficient
	float shininess;			// shininess coefficient
	float reflectance;			// reflectance coefficient
};
typedef struct _mjpMaterial mjpMaterial;


struct _mjpTexture				// texture descriptor
{
	int cube;					// is cube texture (as opposed to 2d)
	int skybox;					// is cube texture used as skybox
	int width;					// width in pixels
	int height;					// height in pixels
	unsigned char* rgb;			// RGB24 texture data
};
typedef struct _mjpTexture mjpTexture;


struct _mjpLight				// light descriptor
{
	int directional;			// is light directional
	int castshadow;				// does ligth cast shadows
	float ambient[3];			// ambient rgb color
	float diffuse[3];			// diffuse rgb color
	float specular[3];			// specular rgb color
	float attenuation[3];		// OpenGL quadratic attenuation model
	float cutoff;				// OpenGL cutoff angle for spot lights
	float exponent;				// OpenGL exponent
};
typedef struct _mjpLight mjpLight;


struct _mjpCamera				// camera descriptor
{
	float fov;					// field of view
	float znear;				// near depth plane
	float zfar;					// far depth plane
	int width;					// offscreen width
	int height;					// offscreen height
};
typedef struct _mjpCamera mjpCamera;


struct _mjpPerturb				// perturbation state (subset of mjvPerturb)
{
    int select;					// selected body id; non-positive: none
    int active;					// bitmask: 1: translation, 2- rotation
    float refpos[3];			// desired position for selected object
    float refquat[4];			// desired orientation for selected object
};
typedef struct _mjpPerturb mjpPerturb;


struct _mjpTransform			// spatial transform
{
	float position[3];			// position
	float xaxis[3];				// x-axis (right in MuJoCo)
	float yaxis[3];				// y-axis (forward in MuJoCo)
	float zaxis[3];				// z-axis (up in MuJoCo)
	float scale[3];				// scaling
};
typedef struct _mjpTransform mjpTransform;


//------------------------------ Initializaion and Simulation ---------------------------

// initialize plugin
MJPAPI int mjp_initialize(void);

// close plugin
MJPAPI int mjp_close(void);

// load model from file in XML (MJCF or URDF) or MJB format
MJPAPI int mjp_loadModel(const char* modelfile);

// save compiled model as MJB file
MJPAPI int mjp_saveMJB(const char* modelfile);

// reset simulation
MJPAPI int mjp_reset(void);

// reset simulation to specified keyframe
MJPAPI int mjp_resetKeyframe(int index);

// compute forward kinematics (sufficient for rendering)
MJPAPI int mjp_kinematics(void);

// advance simulation until time marker is reached or internal reset
MJPAPI int mjp_simulate(float marker, int paused, int* reset);


//------------------------------ Get and Set --------------------------------------------

// (const) get model sizes
MJPAPI int mjp_getSize(mjpSize* size);

// (const) get name of specified renderable object
MJPAPI int mjp_getObjectName(int index, char* buffer, int buffersize);

// (const) get name of specified model element; type is mjpElement
MJPAPI int mjp_getElementName(int type, int index, char* buffer, int buffersize);

// (const) get index of body with specified name; -1: not found
MJPAPI int mjp_getBodyIndex(const char* name, int* index);

// (const) get descriptor of specified renderable object
MJPAPI int mjp_getObject(int index, mjpObject* object);

// (const) get descriptor of specified material
MJPAPI int mjp_getMaterial(int index, mjpMaterial* material);

// (const) get descriptor of specified texture
MJPAPI int mjp_getTexture(int index, mjpTexture* texture);

// (const) get descriptor of specified light; -1: head light
MJPAPI int mjp_getLight(int index, mjpLight* light);

// (const) get descriptor of specified camera; -1: main camera
MJPAPI int mjp_getCamera(int index, mjpCamera* camera);

// get state of specified renderable object
MJPAPI int mjp_getObjectState(int index, mjpTransform* transform, int* visible, int* highlight);

// get state of specified light; -1: head light (use camera index)
MJPAPI int mjp_getLightState(int index, int cameraindex, float* position, float* direction);

// get state of specified camera; -1: main camera
MJPAPI int mjp_getCameraState(int index, mjpTransform* transform);

// get state of specified body relative to parent body
MJPAPI int mjp_getBodyRelativeState(int index, mjpTransform* transform);

// get text description of last error
MJPAPI int mjp_getErrorText(char* buffer, int buffersize);

// get text description of last warning
MJPAPI int mjp_getWarningText(char* buffer, int buffersize);

// get number of warnings since last load or reset
MJPAPI int mjp_getWarningNumber(int* number);

// set system position vector; size(qpos) = nqpos
MJPAPI int mjp_setQpos(const float* qpos);

// set all mocap body poses; size(pos) = 3*nmocap, size(quat) = 4*nmocap
MJPAPI int mjp_setMocap(const float* pos, const float* quat);

// get simulation time
MJPAPI int mjp_getTime(float* time);

// set simulation time
MJPAPI int mjp_setTime(float time);

// get options
MJPAPI int mjp_getOption(mjpOption* option);

// set options
MJPAPI int mjp_setOption(const mjpOption* option);

// get perturbation state
MJPAPI int mjp_getPerturb(mjpPerturb* perturb);

// set perturbation state
MJPAPI int mjp_setPerturb(const mjpPerturb* perturb);


//------------------------------ Camera and Perturbation --------------------------------
// 2D data in relative screen coordinates between 0 and 1

// set main camera lookat point; aspect = width/height
MJPAPI int mjp_cameraLookAt(float x, float y, float aspect);

// zoom main camera
MJPAPI int mjp_cameraZoom(float zoom);

// move main camera lookat point
MJPAPI int mjp_cameraMove(float dx, float dy, int modified);

// rotate main camera around lookat point
MJPAPI int mjp_cameraRotate(float dx, float dy);

// set active bitmask only (use setPerturb for full access)
MJPAPI int mjp_perturbActive(int state);

// set perturb object pose equal to selected body pose
MJPAPI int mjp_perturbSynchronize(void);

// select body for perturbation
MJPAPI int mjp_perturbSelect(float x, float y, float aspect);

// move perturbation object
MJPAPI int mjp_perturbMove(float dx, float dy, int modified);

// rotate perturbation object
MJPAPI int mjp_perturbRotate(float dx, float dy, int modified);


#ifdef __cplusplus
}
#endif
