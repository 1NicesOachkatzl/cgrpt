#include <stdlib.h>     // cgrpt, single-source GLSL path tracer by Reinhold Preiner, 2021. based on smallpt by Kevin Beason
#include <GL/gl.h>      // Make: Windows: g++ -O3 cgrpt2.cpp -o cgrpt -lopengl32 -lgdi32
#include <stdio.h>      //         Linux: g++ -O3 cgrpt2.cpp -o cgrpt -lGL -ldl -lX11
#include <chrono>       // Usage: ./cgrpt <samplesPerPixel> <y-resolution>, e.g.: ./cgrpt 4000 600

#include <math.h>
//-------------------------------------------------------------------------------------------------------------------------------
#define GLSL(...) "#version 430\n" #__VA_ARGS__
#if defined (_WIN32)
    #include <windows.h>
#include <vector>
#include <string>
#include <fstream>

#define getGLProcAdress ((void* (*)(const char*)) GetProcAddress(LoadLibraryW(L"opengl32.dll"), "wglGetProcAddress"))
    void initGL() {
        WNDCLASSA wc = { .style=0x23, .lpfnWndProc=DefWindowProcA, .hInstance=GetModuleHandle(0), .lpszClassName="Core"};
        RegisterClassA(&wc);
        HDC hdc = GetDC(CreateWindowExA(0, wc.lpszClassName, 0, 0, 1<<31, 1<<31, 1<<31, 1<<31, 0, 0, wc.hInstance, 0));
        PIXELFORMATDESCRIPTOR pfd = { .nSize=sizeof(pfd), .nVersion=1, .dwFlags=0x25, .iPixelType=0,
            .cColorBits=32, .cAlphaBits=8, .cDepthBits=24, .cStencilBits=8, .iLayerType=0 };
        SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
        wglMakeCurrent(hdc, wglCreateContext(hdc));
    }
#elif defined (__linux__)       // Requires newest mesa-common-dev, and 'export MESA_GL_VERSION_OVERRIDE=4.5FC'
    #include <dlfcn.h>
    #include <GL/glx.h>
    #define getGLProcAdress ((void* (*)(const char*)) dlsym(libGL, "glXGetProcAddress"))
    static void* libGL = nullptr;
    void initGL() {
        int attr[] = {GLX_RGBA, 0}; libGL = (libGL=dlopen("libGL.so.1", 0x102)) ? libGL : dlopen("libGL.so", 0x102);
        Display *dpy = XOpenDisplay(0); XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
        GLXContext cx = glXCreateContext(dpy, vi, 0, GL_TRUE);
        XSetWindowAttributes swa = { .border_pixel=0, .event_mask=0x28001,
            .colormap=XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, 0) };
        Window win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0,0,300,300,0, vi->depth, 1, vi->visual, 0x2808, &swa);
        glXMakeCurrent(dpy, win, cx);
    }
#endif // other OS not supported
void cbGLError(GLenum, GLenum, GLuint, GLenum, GLsizei, const char* msg, const void*) { fprintf(stderr, "%s\n", msg); exit(0); }
template<class R = void, class ... A> R glFunc (const char* n, A... a) { return ((R (*)(A...)) getGLProcAdress(n)) (a...); };
//-------------------------------------------------------------------------------------------------------------------------------

void setGpuBuffer(int binding, int bufferName, size_t numBytes, void* data = nullptr) {
    glFunc("glNamedBufferData", bufferName, numBytes, data, 0x88E4);     // GL_STATIC_DRAW
    glFunc("glBindBufferBase", 0x90D2, binding, bufferName);    	     // GL_SHADER_STORAGE_BUFFER
}

int createProgram(const char* src) {
    int p = glFunc<int>("glCreateProgram");
    int s = glFunc<int>("glCreateShader", 0x91B9);      // GL_COMPUTE_SHADER
    glFunc("glShaderSource", s, 1, &src, 0);
    glFunc("glCompileShader", s);
    glFunc("glAttachShader", p, s);
    glFunc("glLinkProgram", p);
    glFunc("glUseProgram", p);
    return p;
}

#define BOX_HX	2.6
#define BOX_HY	2
#define BOX_HZ	10

float spheres[] = {  // center.xyz, radius  |  emmission.xyz, 0  |  color.rgb, refltype
    -1e5 - BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  .250, .250, .250,  1, // Left (DIFFUSE)
    1e5  + BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  .250, .250, .250,  1, // Right
    0, 1e5 + BOX_HY, 0, 1e5,        0, 0, 0, 0,  .250, .250, .250,  1, // Top
    0,-1e5 - BOX_HY, 0, 1e5,        0, 0, 0, 0,  .250, .250, .250,  1, // Bottom
    0, 0, -1e5 - BOX_HZ, 1e5,       0, 0, 0, 0,  .250, .250, .250,  1, // Back
    0, 0, 1e5 + 3*BOX_HZ, 1e5,      0, 0, 0, 0,  .250, .250, .250,  1, // Front
    1, 1.5, -1, 0.5, 0, 0, 0, 0,  .2,.20,.20, 1, // occluder geometry
    -1, -1.5, 1, 0.5, 0, 0, 0, 0,  .20,.20,.20, 1, // occluder geometry
    1.5, -1.5, -1, 0.2, 3, 3, 3, 0,  .0,.0,.0, 1,
    -1.5, 1.5, +1, 0.2, 3, 3, 3, 0,  .0,.0,.0, 1,
};

float cornell[] = {
    -1e5 - BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  .750, .250, .250,  1, // Left (DIFFUSE)
    1e5  + BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  .250, .250, .750,  1, // Right
    0, 1e5 + BOX_HY, 0, 1e5,        0, 0, 0, 0,  .750, .750, .750,  1, // Top
    0,-1e5 - BOX_HY, 0, 1e5,        0, 0, 0, 0,  .750, .750, .750,  1, // Bottom
    0, 0, -1e5 - BOX_HZ, 1e5,       0, 0, 0, 0,  .750, .750, .750,  1, // Back
    0, 0, 1e5 + 3*BOX_HZ, 1e5,      0, 0, 0, 0,  .000, .000, .000,  1, // Front
       0, BOX_HY+1.8,   0,   2,    10,   10,  10, 0,  .0, .0, .0,  1, // Light
     1.5,      -0.75, 0.5, 0.2,   200,  100,   0, 0,  .0, .0, .0,  1, // sphere (EMISSIVE)
     -1.5,      0.75, -7, 0.2,   0,  100,   200, 0,  .0, .0, .0,  1, // sphere (EMISSIVE)
      0.5, -0.25, -1, 0.5,             0, 0, 0, 0,  .250, .250, .750, 1, // Glass sphere (DIFFUSE)
    -0.75,     0,  0, 0.5,             0, 0, 0, 0,  .999, .999, .999, 2, // Mirror sphere (SPEC)
     0.75,     1,  0, 0.5,             0, 0, 0, 0,  .999, .999, .999, 3, // Glass sphere (REFR)
};

float bunny_cornell[] = {
    -1e5 - BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  .750, .250, .250,  1, // Left (DIFFUSE)
    1e5  + BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  .250, .250, .750,  1, // Right
    0, 1e5 + BOX_HY, 0, 1e5,        0, 0, 0, 0,  .750, .750, .750,  1, // Top
    0,-1e5 - BOX_HY, 0, 1e5,        0, 0, 0, 0,  .750, .750, .750,  1, // Bottom
    0, 0, -1e5 - BOX_HZ, 1e5,       0, 0, 0, 0,  .750, .750, .750,  1, // Back
    0, 0, 1e5 + 3*BOX_HZ, 1e5,      0, 0, 0, 0,  .000, .000, .000,  1, // Front
       0, BOX_HY+1.8,   0,   2,    10,   10,  10, 0,  .0, .0, .0,  1, // Light
     1.5,      -0.75, 0.5, 0.2,   200,  100,   0, 0,  .0, .0, .0,  1, // sphere (EMISSIVE)
     -1.5,      0.75, -7, 0.2,   0,  100,   200, 0,  .0, .0, .0,  1, // sphere (EMISSIVE)
};

float scene1[] = {
        -1e5 - BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  0, 0, 0,  1, // Left (DIFFUSE)
        1e5  + BOX_HX, 0, 0, 1e5,       0, 0, 0, 0,  0, 0, 0,  1, // Right
        0, 1e5 + BOX_HY, 0, 1e5,        0, 0, 0, 0,  0, 0, 0,  1, // Top
        0,-1e5 - BOX_HY, 0, 1e5,        0, 0, 0, 0,  .75, .75, .75,  1, // Bottom
        0, 0, -1e5 - BOX_HZ, 1e5,       0, 0, 0, 0,  0, 0, 0,  1, // Back
        0, 0, 1e5 + 3*BOX_HZ, 1e5,      0, 0, 0, 0,  0, 0, 0,  1, // Front
        0, -0.5, -1, 0.5, 0, 0, 0, 0,  .999,.999,.999, 1, // occluder geometry
};

int sceneIndex = 0;
float els = 0.0f;

float *vertices;
std::vector<float> verticesVector;

float *vertexIndices;
std::vector<float> vertexIndexVector;
                             //  els   ka  // ks  // kt // g
float renderingParameters[] = {  0,   0.01,      0.1,      0,      0};

struct Vec4{
    float x, y, z, w;
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x, float y, float z, float w = 1.0f) : x(x), y(y), z(z), w(w) {}
};

struct Mat {
    float a0, a1, a2;
    float b0, b1, b2;
    float c0, c1, c2;

    Mat(float a0, float a1, float a2, float b0, float b1, float b2, float c0, float c1, float c2)
        : a0(a0), a1(a1), a2(a2), b0(b0), b1(b1), b2(b2), c0(c0), c1(c1), c2(c2) {}

    Vec4 operator*(const Vec4& v) const {
        return Vec4(
            a0 * v.x + a1 * v.y + a2 * v.z,
            b0 * v.x + b1 * v.y + b2 * v.z,
            c0 * v.x + c1 * v.y + c2 * v.z,
            v.w // Keep the w component unchanged
        );
    }
};

int currentVertexOffset = 0;

void loadOffFile(const char* path, Vec4 geo, Vec4 e, Vec4 c, Vec4 scale, Vec4 rotation)
{
    // calculate rotation matrix
    float cosA = cos(rotation.x), cosB = cos(rotation.y), cosG = cos(rotation.z);
    float sinA = sin(rotation.x), sinB = sin(rotation.y), sinG = sin(rotation.z);

    Mat rot = Mat(  cosA * cosB,    cosA * sinB * sinG - sinA * cosG, cosA * sinB * cosG + sinA * sinG,
                    sinA * cosB,    sinA * sinB * sinG + cosA * cosG, sinA * sinB*cosG - cosA * sinG,
                    -sinB,           cosB * sinG,                      cosB * cosG);

    std::ifstream inFile;
    inFile.open(path);
    std::string S;
    std::string FileType;

    int numVertices, numFaces, numCells;
    inFile >> FileType >> numVertices>>numFaces>>numCells;
    Vec4 vertex;

    // load vertices into vertices list
    for(int i = 0; i < numVertices; i++)
    {
        inFile>>vertex.x>>vertex.y>>vertex.z;

        // scale
        vertex.x *= scale.x;
        vertex.y *= scale.y;
        vertex.z *= scale.z;

        // translation
        vertex.x += geo.x;
        vertex.y += geo.y;
        vertex.z += geo.z;
        vertex.w = 0;

        // geo rot scale vert  e   c
        verticesVector.insert(verticesVector.end(), {geo.x, geo.y, geo.z, geo.w});
        verticesVector.insert(verticesVector.end(), {rot.a0, rot.a1, rot.a2, 0});
        verticesVector.insert(verticesVector.end(), {rot.b0, rot.b1, rot.b2, 0});
        verticesVector.insert(verticesVector.end(), {rot.c0, rot.c1, rot.c2, 0});
        verticesVector.insert(verticesVector.end(), {scale.x, scale.y, scale.z, 0});
        verticesVector.insert(verticesVector.end(), {vertex.x, vertex.y, vertex.z, vertex.w});
        verticesVector.insert(verticesVector.end(), {e.x, e.y, e.z, e.w});
        verticesVector.insert(verticesVector.end(), {c.x, c.y, c.z, c.w});
    }

    double numTriangles, vertIndex1, vertIndex2, vertIndex3, vertIndex4;

    // load indices into indices list
    for(int i = 0; i < numFaces; i++)
    {
        inFile>>numTriangles>>vertIndex1>>vertIndex2>>vertIndex3;

        vertexIndexVector.emplace_back(vertIndex1 + currentVertexOffset);
        vertexIndexVector.emplace_back(vertIndex2 + currentVertexOffset);
        vertexIndexVector.emplace_back(vertIndex3 + currentVertexOffset);
        vertexIndexVector.emplace_back(0);

        if(numTriangles == 4)
        {
            inFile>>vertIndex4;
            vertexIndexVector.emplace_back(vertIndex1 + currentVertexOffset);
            vertexIndexVector.emplace_back(vertIndex3 + currentVertexOffset);
            vertexIndexVector.emplace_back(vertIndex4 + currentVertexOffset);
            vertexIndexVector.emplace_back(0);
        }

    }
    currentVertexOffset += numVertices;
}

void loadMeshes()
{
    const char* fileName = "meshes/pyramid.off";

    if(els > 0.1)
    {
        loadOffFile(fileName, Vec4{0,1,-1,0}, Vec4{.5f*8,0.5f*8,0.5f*8,0},
                    Vec4{0,0,0,1}, Vec4{0.005,0.005,0.005,0},
                    Vec4{0,0,0,0});
    }
    else
    {
        loadOffFile(fileName, Vec4{0,1,-1,0}, Vec4{2*2,2*2,2*2,0},
                    Vec4{0,0,0,1}, Vec4{0.005,0.005,0.005,0},
                    Vec4{0,0,0,0});
    }
}

int main(int argc, char *argv[]) {
    //-- parameter info
	if (argc >= 2 && *argv[1] == '?') {
        printf("./cgrpt <samplesPerPixel = 4000> <y-resolution = 600>\n");
        exit (0);
    }

    initGL();
    //glEnable(0x92E0); glFunc("glDebugMessageCallback", cbGLError, 0);     // activate for GL debugging

    //-- parse arguments
    int spp = argc>1 ? atoi(argv[1]) : 1000;    // samples per pixel
    int resy = argc>2 ? atoi(argv[2]) : 600;    // vertical pixel resolution
    int resx = resy*3/2;	                    // horizontal pixel resolution

    sceneIndex = argc>3 ? atoi(argv[3]) : 0;

    els = argc>4 ? atoi(argv[4]) : 0.0f;
    float ka = argc>5 ? atof(argv[5]) : 0.0f;
    float ks = argc>6 ? atof(argv[6]) : 0.0f;
    float g = argc>7 ? atof(argv[7]) : 0.0f;

    renderingParameters[0] = els;
    renderingParameters[1] = ka;
    renderingParameters[2] = ks;
    renderingParameters[3] = ka + ks;
    renderingParameters[4] = g;

    // load meshes for scene 1 or scene 2
    if(sceneIndex == 1)
        loadMeshes();


    vertices = reinterpret_cast<GLfloat *>(verticesVector.data());
    vertexIndices = reinterpret_cast<GLfloat *>(vertexIndexVector.data());

    //-- create data buffers
    struct { int Sph, Rad, Vert, VertIdx, aMapParams; } buf;
    glFunc("glCreateBuffers", sizeof(buf)/sizeof(int), &buf);

    if(sceneIndex == 0)
    {
        setGpuBuffer(0, buf.Sph, sizeof(cornell), cornell);
    }
    else if(sceneIndex == 1)
    {
        setGpuBuffer(0, buf.Sph, sizeof(spheres), spheres);
    }


    setGpuBuffer(1, buf.Rad, 4*resx*resy*sizeof(float));        // binding 1: pixel buffer accumulating radiance
    setGpuBuffer(2, buf.Vert, verticesVector.size() * sizeof(float), vertices);
    setGpuBuffer(3, buf.VertIdx, vertexIndexVector.size() * sizeof(float), vertexIndices);
    setGpuBuffer(4, buf.aMapParams, sizeof(renderingParameters), renderingParameters);

    //-- create GLSL program
    int p = createProgram(GLSL(
        layout(local_size_x = 16, local_size_y = 16) in;
        struct Ray { vec3 o; vec3 d; };
        struct Sphere { vec4 geo; vec3 e; vec4 c; };
        struct Vertex {vec4 geo; mat3 rot; vec3 scale; vec3 v; vec3 e; vec4 c;}verObj;
        struct VertexIndex {vec4 idx;}verIdxObj;
        struct RenderingParameters{float els; float ka; float ks; float kt; float g;};

        layout(std430, binding = 0) buffer b0 { Sphere spheres[]; };
        layout(std430, binding = 1) buffer b1 { vec4 accRad[]; };
        layout(std430, binding = 2) buffer b2 { Vertex vertices[]; };
        layout(std430, binding = 3) buffer b3 { VertexIndex vertexIndices[]; };
        layout(std430, binding = 4) buffer b4 { RenderingParameters renderingParameters[]; };
        uniform uvec2 imgdim, samps;            // image dimensions and sample count

        const float inf = 1e20;
        const float pi = 3.141592653589793;
        const float eps = 1e-4;
        vec3 rand01(uvec3 x){                   // pseudo-random number generator
            for (int i=3; i-->0;) x = ((x>>8U)^x.yzx)*1103515245U;
            return vec3(x)*(1.0/float(0xffffffffU));
        }

        float intersect(Ray r, out int id, out vec3 x, out vec3 n, out Sphere obj) {
            float d, t = inf;   // intersect ray with scene
            // sphere check
            for (int i = spheres.length(); i-->0;) {
                Sphere s = spheres[i];                  // perform intersection test in double precision
                dvec3 oc = dvec3(s.geo.xyz) - r.o;      // Solve t^2*d.d + 2*t*(o-s).d + (o-s).(o-s)-r^2 = 0
                double b=dot(oc,r.d), det=b*b-dot(oc,oc)+s.geo.w*s.geo.w;
                if (det < 0) continue; else det=sqrt(det);
                d = (d = float(b-det))>eps ? d : ((d=float(b+det))>eps ? d : inf);
                if(d < t) { t=d; id = i; obj = s;}
            }
            if (t < inf) {
                x = r.o + r.d*t;
                n = normalize(x-spheres[id].geo.xyz);
            }

            // triangle check
            for (int i = vertexIndices.length(); i--> 0;)
            {
                // get vertices of triangle using the index buffer
                ivec3 indices = ivec3(vertexIndices[i].idx.xyz);
                vec3 v0 = vertices[indices.x].v;
                vec3 v1 = vertices[indices.y].v;
                vec3 v2 = vertices[indices.z].v;

                vec3 E1 = v1 - v0;
                vec3 E2 = v2 - v0;

                vec3 S = r.o - v0;
                vec3 S1 = cross(r.d, E2);
                vec3 S2 = cross(S, E1);

                float factor = 1.0f / dot(E1, S1);
                float b1 = factor * dot(S1, S);
                float b2 = factor * dot(r.d, S2);

                if (b1 < 0.0 || b1 > 1.0) continue;
                if (b2 < 0.0 || b1 + b2 > 1.0) continue;

                float d = factor * dot(E2, S2);


                if (d > eps && d < t)
                {
                    obj.c = vertices[indices.x].c;
                    t = d;
                    obj.e = vertices[indices.x].e;
                    obj.geo.xyz = vertices[indices.x].geo.xyz;
                }
            }
            return t;
        }

        vec3 DirectIllumination(vec3 x, vec3 rnd, vec3 nl, vec3 n, vec3 accmat, float fp)
        {
            vec3 accrad_tmp = vec3(0,0,0);
            // Direct Illumination: Next Event Estimation over any present lights
            for (int i = spheres.length(); i-- > 0;)
            {
                Sphere ls = spheres[i];
                if (all(equal(ls.e, vec3(0)))) continue; // skip non-emissive spheres
                vec3 xls, nls, xc = ls.geo.xyz - x;
                vec3 sw = normalize(xc), su = normalize(cross((abs(sw.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), sw)),
                     sv = cross(sw, su);
                float cos_a_max = sqrt(float(1 - ls.geo.w * ls.geo.w / dot(xc, xc)));
                float cos_a = 1 - rnd.x + rnd.x * cos_a_max, sin_a = sqrt(1 - cos_a * cos_a);
                float phi = 2 * pi * rnd.y;
                // sampled direction towards light
                vec3 l = normalize(su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a);
                int idls = 0;
                Sphere tmp;
                float ts = intersect(Ray(x, l), idls, xls, nls, tmp);
                if (ts < inf && idls == i) // test if shadow ray hits this light source
                {
                    float omega = 2 * pi * (1 - cos_a_max);

                    if(nl == vec3(0,0,0))
                    {
                        float tau = exp(-renderingParameters[0].kt * length(xls - x));
                        accrad_tmp += (renderingParameters[0].ks * fp * tau * accmat * ls.e * omega / pi)
                                      / renderingParameters[0].kt;
                    }
                    else
                    {
                        accrad_tmp += accmat / pi * max(dot(l, nl), 0) * ls.e * omega;
                    }
                }
            }

            // triangle lights
            for (int i = vertexIndices.length(); i--> 0;) {
                ivec3 indices = ivec3(vertexIndices[i].idx.xyz);
                vec3 v0 = vertices[indices.x].v;
                vec3 v1 = vertices[indices.y].v;
                vec3 v2 = vertices[indices.z].v;
                double eps1 = rnd.x, eps2 = rnd.y, sqrt_eps1 = sqrt(eps1);
                vec3 barycentric = vec3(1 - sqrt_eps1, (1 - eps2) * sqrt_eps1, eps2 * sqrt_eps1);
                vec3 x_prime = v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z;

                float A = length(cross((v1 - v0), (v2 - v0))) / 2;
                vec3 sw = normalize(x_prime - x);

                double tls;
                vec3 xls, nls;
                int idls = 0;
                Sphere tmp;
                float ts = intersect(Ray(x, sw), idls, xls, nls, tmp);
                if (ts < inf && idls == i) // test if shadow ray hits this light source
                {
                    float cos_theta_prime = max(dot(-sw, nls),0);
                    float sqr_dist = dot(x_prime - x, x_prime -x);

                    if(nl == vec3(0,0,0))
                    {
                        float tau = exp(-renderingParameters[0].kt * length(sw));
                        accrad_tmp += (renderingParameters[0].ks * fp * tau * accmat * max(dot(sw, nl), 0) *
                                       vertices[indices.x].e * A * cos_theta_prime / sqr_dist) / renderingParameters[0].kt;
                    }
                    else
                    {
                        accrad_tmp += accmat * max(dot(sw, nl), 0) * vertices[indices.x].e * A * cos_theta_prime / sqr_dist;
                    }
                }
            }
            return accrad_tmp;
        }

        void main() {
            uvec2 pix = gl_GlobalInvocationID.xy;
            if (pix.x >= imgdim.x || pix.y >= imgdim.y) return;
            uint gid = (imgdim.y - pix.y - 1) * imgdim.x + pix.x;

            //-- define camera
            Ray cam = Ray(vec3(0, 0.52, 7.4), normalize(vec3(0, -0.06, -1)));
            vec3 cx = normalize(cross(cam.d, abs(cam.d.y)<0.9 ? vec3(0,1,0) : vec3(0,0,1))), cy = cross(cx, cam.d);
            const vec2 sdim = vec2(0.036, 0.024);    // sensor size (36 x 24 mm)

            //-- sample sensor
            vec2 rnd2 = 2*rand01(uvec3(pix, samps.x)).xy;   // vvv tent filter sample
            vec2 tent = vec2(rnd2.x<1 ? sqrt(rnd2.x)-1 : 1-sqrt(2-rnd2.x), rnd2.y<1 ? sqrt(rnd2.y)-1 : 1-sqrt(2-rnd2.y));
            vec2 s = ((pix + 0.5 * (0.5 + vec2((samps.x/2)%2, samps.x%2) + tent)) / vec2(imgdim) - 0.5) * sdim;
            vec3 spos = cam.o + cx*s.x + cy*s.y, lc = cam.o + cam.d * 0.035;           // sample on 3d sensor plane
            vec3 accrad=vec3(0), accmat=vec3(1);        // initialize accumulated radiance and bxdf
            Ray r = Ray(lc, normalize(lc - spos));      // construct ray

            //-- loop over ray bounces
            float emissive = 1;
            for (int depth = 0, maxDepth = 12; depth < maxDepth; depth++)
            {
                int id = 0;
                vec3 x, n;
                Sphere obj;
                vec3 rnd = rand01(uvec3(pix, samps.x * maxDepth + depth));    // vector of random numbers for sampling

                float t = intersect(r, id, x, n, obj);
                if (t < inf) {   // intersect ray with scene
                    vec3 rnd = rand01(uvec3(pix, samps.x*maxDepth + depth));    // vector of random numbers for sampling

                    //// standard ray sampling
                    if(renderingParameters[0].els < eps)
                    {
                        vec3 x = r.o + r.d * float(t), n = normalize(x - obj.geo.xyz), nl = dot(n, r.d) < 0 ? n : -n;
                        float p = max(max(obj.c.x, obj.c.y), obj.c.z);  // max reflectance
                        accrad += accmat * obj.e;
                        accmat *= obj.c.xyz;
                        vec3 rdir = reflect(r.d, n);

                        if (depth > 5)
                        {
                            if (rnd.z >= p) break;  // Russian Roulette ray termination
                            else accmat /= p;       // Energy compensation of surviving rays
                        }
                        //-- Ideal DIFFUSE reflection
                        if (obj.c.w == 1)
                        {
                            float r1 = 2 * pi * rnd.x, r2 = rnd.y, r2s = sqrt(r2);  // cosine-weighted importance sampling
                            vec3 w = nl, u = normalize((cross(abs(w.x) > 0.1 ? vec3(0, 1, 0) : vec3(1, 0, 0), w))),
                            v = cross(w, u);
                            r = Ray(x, normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)));
                        }
                            //-- Ideal SPECULAR reflection
                        else if (obj.c.w == 2)
                        {
                            r = Ray(x, rdir);
                        }
                            //-- Ideal dielectric REFRACTION
                        else if (obj.c.w == 3)
                        {
                            bool into = n == nl;
                            float cos2t, nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.d, nl);
                            if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) >= 0) // Fresnel reflection/refraction
                            {
                                vec3 tdir = normalize(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))));
                                float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(tdir, n));
                                float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = 0.25 + 0.5 * Re,
                                      RP = Re / P, TP = Tr / (1 - P);
                                r = Ray(x, rnd.x < P ? rdir : tdir);    // pick reflection with probability P
                                accmat *= rnd.x < P ? RP : TP;         // energy compensation
                            }
                            else r = Ray(x, rdir); // Total internal reflection
                        }
                        continue;
                    }

                    //// next event estimation
                    vec3 attenuation_r = rand01(uvec3(pix, samps.x * maxDepth + depth));
                    float s = -log(attenuation_r.x) / renderingParameters[0].kt;

                    // Ray made it through the medium â†’ evaluate surface BxDF
                    if(s >= t)
                    {
                        vec3 nl = dot(n, r.d) < 0 ? n : -n;
                        accrad += accmat * obj.e * emissive;
                        accmat *= obj.c.xyz;
                        // russian roulette
                        // directly link the absorption to the termination probability
                        float p = 1 - renderingParameters[0].ka;
                        if (depth > 5){
                            if (rnd.z >= p) break;  // Russian Roulette ray termination
                            else accmat /= p;       // Energy compensation of surviving rays
                        }
                        //-- Ideal DIFFUSE reflection
                        if (obj.c.w == 1) {
                            // direct illumination
                            accrad += DirectIllumination(x, rnd, nl, n, accmat, 0f);

                            // Indirect Illumination: cosine-weighted importance sampling
                            float r1 = 2 * pi * rnd.x, r2 = rnd.y, r2s = sqrt(r2);
                            vec3 w = nl, u = normalize((cross(abs(w.x) > 0.1 ? vec3(0, 1, 0) : vec3(1, 0, 0), w))),
                                 v = cross(w,u);
                            r = Ray(x, normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)));
                            emissive = 0;   // in the next bounce, consider reflective part only!
                        }
                        //-- Ideal SPECULAR reflection
                        else if (obj.c.w == 2) {
                            r = Ray(x, reflect(r.d, n));
                            emissive = 1;
                        }
                        //-- Ideal dielectric REFRACTION
                        else if (obj.c.w == 3)
                        {
                            bool into = n == nl;
                            float cos2t, nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.d, nl);
                            if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) >= 0) {     // Fresnel reflection/refraction
                                vec3 tdir = normalize(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))));
                                float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(tdir, n));
                                float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re,
                                      P = 0.25 + 0.5 * Re, RP = Re / P, TP = Tr / (1 - P);
                                r = Ray(x, rnd.x < P ? reflect(r.d, n) : tdir);      // pick reflection with probability P
                                accmat *= rnd.x < P ? RP : TP;                       // energy compensation
                            } else r = Ray(x, reflect(r.d, n));                      // Total internal reflection
                            emissive = 1;
                        }
                    }
                    // Ray scatters at ð‘¥ð‘  = x âˆ’ðœ” * s
                    // s < t
                    else
                    {
                        // russian roulette
                        // directly link the absorption to the termination probability
                        float p = 1 - renderingParameters[0].ka;
                        if (depth > 5){
                            if (rnd.z >= p) break;  // Russian Roulette ray termination
                            else accmat /= p;       // Energy compensation of surviving rays
                        }

////                        isotropic phase function and random sampled direction
//                        float fp = 1 / (4 * pi);
//
//                        // generate random direction lecture 8 slide 30
//                        float phi = 2 * pi * rnd.x;
//                        float theta = acos(rnd.y);
//                        vec3 dir;
//                        dir.x = cos(phi) * sqrt(1 - rnd.y * rnd.y);
//                        dir.y = sin(phi) * sqrt(1 - rnd.y * rnd.y);
//                        dir.z = rnd.x * (rnd.z > 0.5? 1: -1); // 50/50 chance for the sign to cover the full hemisphere

                        // Henyey Greenstein
                        // as described in:
                        // https://pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering
                        //       #fragment-ComputecosthetaforHenyey--Greensteinsample-0
                        float fp, cosTheta, g = renderingParameters[0].g;
                        vec3 dir, phase_rnd = rand01(uvec3(samps.x << 8, samps.y << 4, 0));

                        if(abs(g) < eps) { cosTheta = 1 - 2 * phase_rnd.x; }
                        else
                        {
                            float sqrTerm = (1 - g * g) / (1 - g + 2 * g * phase_rnd.x);
                            cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (- 2 * g);
                        }
                        float sinTheta = sqrt(max(0, 1 - cosTheta * cosTheta));
                        float phi = 2 * pi * phase_rnd.y;

                        vec3 v1 = r.d;
                        vec3 v2 = vec3(0,0,0);
                        vec3 v3 = vec3(0,0,0);

                        // generate coordinate system
                        if(abs(v1.x) > abs(v1.y)) {v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z); }
                        else { v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z); }
                        v3 = cross(v1, v2);

                        float denom = 1 + g * g + 2 * g * cosTheta;
                        fp =  (1 / (4 * pi)) * (1 - g * g) / (denom * sqrt(denom));

                        vec3 xs = r.o + s * r.d;
                        dir = sinTheta * cos(phi) * v1 + sinTheta * sin(phi) * v2 + cosTheta * v3;

                        accrad += DirectIllumination(xs, rnd, vec3(0,0,0), n, accmat, fp);
                        r = Ray(xs, dir);
                        emissive = 0;
                    }
                }
            }

            if (samps.x == 0) accRad[gid] = vec4(0);    // initialize radiance buffer
                accRad[gid] += vec4(accrad / samps.y, 0);   // <<< accumulate radiance   vvv write 8bit rgb gamma encoded color
            if (samps.x == samps.y-1) accRad[gid].xyz = pow(vec3(clamp(accRad[gid].xyz, 0, 1)), vec3(0.45)) * 255 + 0.5;
        }
    ));

    //-- sample rays
    auto tstart = std::chrono::system_clock::now();		                    // take start time
    glFunc("glUniform2ui", glFunc<int>("glGetUniformLocation", p, "imgdim"), resx, resy);
    for (int pass = 0; pass < spp; pass++) {
        glFunc("glUniform2ui", glFunc<int>("glGetUniformLocation", p, "samps"), pass, spp);
        glFunc("glDispatchCompute", (resx + 15) / 16, (resy + 15) / 16, 1);
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", spp, 100.0 * (pass+1) / spp);
        glFinish();
    }
    auto tend = std::chrono::system_clock::now();

    //-- read back buffer and write inverse sensor image to file
    float* pixels = new float[4*resx*resy];
    glFunc("glGetNamedBufferSubData", buf.Rad, 0, 4*resx*resy*sizeof(float), pixels);
    FILE *file = fopen("image.ppm", "w");
    fprintf(file, "P3\n# spp: %d\n", spp);
    fprintf(file, "# time: %.2f s\n", std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
    fprintf(file, "%d %d %d\n", resx, resy, 255);
    for (int i = resx*resy; i-->0;)
        fprintf(file, "%d %d %d ", int(pixels[4*i]), int(pixels[4*i+1]), int(pixels[4*i+2]));
}
