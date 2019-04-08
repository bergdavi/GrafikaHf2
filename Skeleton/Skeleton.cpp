//=============================================================================================
// Mintaprogram: Zold haromszag. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Berghammer David
// Neptun : EB2DYD
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec2 vp;

    uniform vec3 cameraLookAt;
    uniform vec3 cameraRight;
    uniform vec3 cameraUp;

    out vec3 p;

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1);
        p = cameraLookAt + cameraRight*vp.x + cameraUp*vp.y;
	}
)";

const char * const fragmentSource = R"(
#version 330
precision highp float;

struct Light {
    vec3 direction;
    vec3 Le, La;
};

struct Material {
    vec3 ka, kd, ks, n, k;
    float shininess;
    int reflective;
    int rough;
};

struct Triangle {
    vec4 r1, r2, r3, n;
    int material;
};

struct Ellipsoid {
    vec4 center;
    vec3 size;
    int material;
};

struct Hit {
    float t;
    vec3 position, normal;
    int material;
};

struct Ray {
    vec3 start, dir;
};

const int maxObjects = 50;

uniform vec3 cameraEye;
uniform Light light;
uniform Material materials[5];
uniform Ellipsoid ellipsoids[maxObjects];
uniform Triangle triangles[maxObjects];

uniform int ellipsoidCount;
uniform int triangleCount;

in vec3 p;
out vec4 outColor;


Hit intersect(const Ellipsoid e,const Ray ray) {
    Hit hit;
    hit.t = -1;

    mat4 translate = mat4(
        1/e.size.x, 0         , 0         , 0,
        0         , 1/e.size.y, 0         , 0,
        0         , 0         , 1/e.size.z, 0,
        0         , 0         , 0         , 1
    );

    mat4 translateInv = mat4(
        e.size.x, 0       , 0       , 0,
        0       , e.size.y, 0       , 0,
        0       , 0       , e.size.z, 0,
        0       , 0       , 0       , 1
    );

    float radius = 1;

    vec3 center = (e.center*translate).xyz;
    vec3 start = (vec4(ray.start, 1)*translate).xyz;
    vec3 dir = (vec4(ray.dir, 1)*translate).xyz;

    vec3 dist = start-center;
    float a = dot(dir, dir);
    float b = dot(dist, dir)*2;
    float c = dot(dist, dist) - radius*radius;
    float discr = b*b-4*a*c;
    if(discr < 0) {
        return hit;
    }
    discr = sqrt(discr);
    float t1 = (-b+discr)/2/a;
    float t2 = (-b-discr)/2/a;
    if(t1 <= 0) {
        return hit;
    }
    float t = (t2 > 0)?t2:t1;
    vec3 position = start + dir*t;
    vec3 normal = position - center;

    hit.position = (vec4(position, 1)*translateInv).xyz;
    hit.normal = normalize((vec4(normal, 1)*translate).xyz);
    hit.t = length(hit.position-ray.start);

    hit.material = e.material;
    return hit;
}


Hit intersect(const Triangle tri, const Ray ray) {
    Hit hit;
    hit.t = -1;

    float t = dot(tri.r1.xyz - ray.start, tri.n.xyz) / dot(ray.dir, tri.n.xyz);
    if(t < 0) {
        return hit;
    }
    vec3 p = ray.start + ray.dir * t;


    if(dot(cross(tri.r2.xyz-tri.r1.xyz, p-tri.r1.xyz), tri.n.xyz) < 0) {
        return hit;
    }

    if(dot(cross(tri.r3.xyz-tri.r2.xyz, p-tri.r2.xyz), tri.n.xyz) < 0) {
        return hit;
    }

    if(dot(cross(tri.r1.xyz-tri.r3.xyz, p-tri.r3.xyz), tri.n.xyz) < 0) {
        return hit;
    }


    hit.t = t;
    hit.position = p;
    hit.normal = tri.n.xyz;
    hit.material = tri.material;

    return hit;
}

Hit firstIntersect(Ray ray) {
    Hit bestHit;
    bestHit.t = -1;
    for (int i = 0; i < ellipsoidCount; i++) {
        Hit hit = intersect(ellipsoids[i], ray);
        if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
    }
    for (int i = 0; i < triangleCount; i++) {
        Hit hit = intersect(triangles[i], ray);
        if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
    }
    if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
    return bestHit;
}

bool shadowIntersect(Ray ray) {
    for (int i = 0; i < ellipsoidCount; i++) {
        if (intersect(ellipsoids[i], ray).t > 0) return true;
    }
    for (int i = 0; i < triangleCount; i++) {
        if (intersect(triangles[i], ray).t > 0) return true;
    }
    return false;
}

vec3 fresnel(vec3 F0, float cosTheta) {
    return F0 + (vec3(1, 1, 1) - F0) * pow(1-cosTheta, 5);
}

vec3 fresnel(vec3 vi, vec3 vn, vec3 n, vec3 kappa) {
    float cosa = dot(-vi, vn);
    vec3 v1 = vec3(1,1,1);
    vec3 F0 = ((n-v1)*(n-v1)+kappa*kappa)/((n+v1)*(n+v1)+kappa*kappa);
    return fresnel(F0, cosa);
}

const float epsilon = 0.0001f;
const int maxDepth = 20;

vec3 trace(Ray r) {
    vec3 outRadiance = vec3(0,0,0);
    vec3 weight = vec3(1,1,1);

    for(int d = 0; d < maxDepth; d++) {
        Hit hit = firstIntersect(r);
        if(hit.t < 0) {
            outRadiance += weight*light.La;
            break;
        }
        if(materials[hit.material].rough == 1) {
            outRadiance += weight*materials[hit.material].ka*light.La;
            vec3 pos = hit.position + hit.normal * epsilon;
            vec3 lightDirection = light.direction;
            Ray shadowRay;
            shadowRay.start = pos;
            shadowRay.dir = -light.direction;
            float cosTheta = dot(hit.normal, -lightDirection);
            if(cosTheta > 0 && !shadowIntersect(shadowRay)) {
                outRadiance += weight * light.Le*materials[hit.material].kd * cosTheta;
                vec3 halfway = normalize(-r.dir - lightDirection);
                float cosDelta = dot(hit.normal, halfway);
                if(cosDelta > 0) {
                    outRadiance += weight * light.Le*materials[hit.material].ks*pow(cosDelta, materials[hit.material].shininess);
                }
            }
        }
        if(materials[hit.material].reflective == 1) {
            weight *= fresnel(r.dir, hit.normal, materials[hit.material].n, materials[hit.material].k);
            r.start = hit.position + hit.normal * epsilon;
            r.dir = reflect(r.dir, hit.normal);
        }
        else {
            break;
        }
    }
    return outRadiance;
}

void main() {
    Ray ray;
    ray.start = cameraEye;
    ray.dir = normalize(p - cameraEye);
    outColor = vec4(trace(ray), 1);
}
)";

struct Camera {
    vec3 eye, lookAt, right, up;

    Camera(const vec3 &eye, const vec3 &lookAt, const vec3 &vup, double fov) : eye(eye), lookAt(lookAt) {
        vec3 w = eye - lookAt;
        float f = length(w);
        right = normalize(cross(vup, w)) * f *tan(fov / 2);
        up = normalize(cross(w, right)) * f * tan(fov / 2);
    }

    void setUniform(unsigned int shaderProgram) {
        char buffer[100];

        sprintf(buffer, "cameraEye");
        eye.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "cameraLookAt");
        lookAt.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "cameraRight");
        right.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "cameraUp");
        up.SetUniform(shaderProgram, buffer);
    }
};

struct Light {
    vec3 direction, Le, La;

    Light(const vec3 &direction, const vec3 &Le, const vec3 &La) : direction(normalize(direction)), Le(Le), La(La) {}

    void setUniform(unsigned int shaderProgram) {
        char buffer[100];

        sprintf(buffer, "light.direction");
        direction.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "light.Le");
        Le.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "light.La");
        La.SetUniform(shaderProgram, buffer);
    }
};

struct Material {
    vec3 ka, kd, ks, n, k;
    float shininess;
    bool reflective = false;
    bool rough = false;

    Material(const vec3 &kd, const vec3 &ks, const vec3 &n, const vec3 &k, float shininess) : ka(kd*M_PI), kd(kd), ks(ks), n(n), k(k), shininess(shininess) {}

    void setUniform(unsigned int shaderProgram, int idx) {
        char buffer[100];
        int location = -1;

        sprintf(buffer, "materials[%d].ka", idx);
        ka.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "materials[%d].kd", idx);
        kd.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "materials[%d].ks", idx);
        ks.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "materials[%d].n", idx);
        n.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "materials[%d].k", idx);
        k.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "materials[%d].shininess", idx);
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1f(location, shininess); else printf("uniform %s cannot be set\n", buffer);

        sprintf(buffer, "materials[%d].reflective", idx);
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform %s cannot be set\n", buffer);

        sprintf(buffer, "materials[%d].rough", idx);
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform %s cannot be set\n", buffer);
    }
};

struct Triangle {
    vec4 r1, r2, r3, n;
    int material;

    Triangle(const vec3 &r1, const vec3 &r2, const vec3 &r3, int material) : r1(vec4(r1.x, r1.y, r1.z, 1)), r2(vec4(r2.x, r2.y, r2.z, 1)), r3(vec4(r3.x, r3.y, r3.z, 1)), material(material) {
        vec3 n3 = normalize(cross(r2 - r1, r3 - r1));
        n = vec4(n3.x, n3.y, n3.z, 1);
    }

    void setUniform(unsigned int shaderProgram, int idx) {
        char buffer[100];
        int location = -1;

        sprintf(buffer, "triangles[%d].r1", idx);
        r1.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "triangles[%d].r2", idx);
        r2.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "triangles[%d].r3", idx);
        r3.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "triangles[%d].n", idx);
        n.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "triangles[%d].material", idx);
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1i(location, material); else printf("uniform %s cannot be set\n", buffer);
    }
};

struct Ellipsoid {
    vec4 center;
    vec3 size;
    int material;

    Ellipsoid(const vec4 &center, const vec3 &size, int material) : center(center), size(size), material(material) {}

    void setUniform(unsigned int shaderProgram, int idx) {
        char buffer[100];
        int location = -1;

        sprintf(buffer, "ellipsoids[%d].center", idx);
        center.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "ellipsoids[%d].size", idx);
        size.SetUniform(shaderProgram, buffer);

        sprintf(buffer, "ellipsoids[%d].material", idx);
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1i(location, material); else printf("uniform %s cannot be set\n", buffer);
    }
};

class Scene {
    std::vector<Ellipsoid*> ellipsoids;
    std::vector<Triangle*> triangles;
    std::vector<Material*> materials;
    Camera *camera;
    Light *light;
    vec3 center;

public:

    int sides = 3;
    int material = 0;

    void build() {
        center = vec3(0, 0, 11);
        camera = new Camera(vec3(0, 0, 0), vec3(0, 0, 10), vec3(0, 1, 0), 80 * M_PI / 180.0);
        light = new Light(vec3(0.5, -1, 0), vec3(10, 10, 10), vec3(0.9, 0.9, 0.9));

        Material* gold = new Material(vec3(0.166f, 0.138f, 0.044f), vec3(0.5, 0.5, 0.5), vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9), 50);
        gold->reflective = true;
        gold->rough = false;

        Material* silver = new Material(vec3(0.15f, 0.15f, 0.15f), vec3(0.5, 0.5, 0.5), vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1), 50);
        silver->reflective = true;
        silver->rough = false;

        Material * red = new Material(vec3(0.3f, 0, 0), vec3(0.5, 0.5, 0.5), vec3(0, 0, 0), vec3(0, 0, 0), 50);
        red->rough = true;

        Material * yellow = new Material(vec3(0.3f, 0.3f, 0), vec3(0.5, 0.5, 0.5), vec3(0, 0, 0), vec3(0, 0, 0), 500);
        yellow->rough = true;

        Material * blue = new Material(vec3(0, 0.3f, 0.3f), vec3(0.5, 0.5, 0.5), vec3(0, 0, 0), vec3(0, 0, 0), 500);
        blue->rough = true;

        materials.push_back(gold); materials.push_back(silver); materials.push_back(red); materials.push_back(yellow); materials.push_back(blue);

        buildTriangles();

        ellipsoids.push_back(new Ellipsoid(vec4(-0.5, 0, 11, 1), vec3(0.15, 0.25, 0.1), 2));
        ellipsoids.push_back(new Ellipsoid(vec4(0.5, 0, 11, 1), vec3(0.15, 0.15, 0.15), 3));
        ellipsoids.push_back(new Ellipsoid(vec4(0, 0.5, 11, 1), vec3(0.15, 0.1, 0.1), 4));
    }

    void buildTriangles() {
        triangles.clear();
        vec4 v = vec4(0, 1, 0, 1);
        vec4 o = vec4(0, 0, 0, 1);


        mat4 rotMx = RotationMatrix(2 * M_PI / sides, vec3(0, 0, 1));

        v = v * RotationMatrix(M_PI / sides, vec3(0, 0, 1));

        vec4 lastP = o + v;

        for (int i = 0; i < sides; i++) {
            v = v * rotMx;
            vec4 p = o + v;

            Triangle* t1 = new Triangle(vec3(lastP.x, lastP.y, 0), vec3(p.x, p.y, 0), vec3(lastP.x, lastP.y, 10), material);
            Triangle* t2 = new Triangle(vec3(lastP.x, lastP.y, 10), vec3(p.x, p.y, 10), vec3(p.x, p.y, 0), material);

            triangles.push_back(t1);
            triangles.push_back(t2);

            lastP = p;
        }
    }

    void animate(float dt) {
        for (int i = 0; i < ellipsoids.size(); i++) {
            ellipsoids[i]->center = ellipsoids[i]->center + randomForce()*dt;

            vec3 eCenter(ellipsoids[i]->center.x, ellipsoids[i]->center.y, ellipsoids[i]->center.z);

            if (length(eCenter - center) > 1) {
                eCenter = center + normalize(eCenter - center);
                ellipsoids[i]->center.x = eCenter.x;
                ellipsoids[i]->center.y = eCenter.y;
                ellipsoids[i]->center.z = eCenter.z;
            }
        }
    }

    vec4 randomForce() {
        return vec4((float)rand() / (float)RAND_MAX * 2 - 1, (float)rand() / (float)RAND_MAX * 2 - 1, (float)rand() / (float)RAND_MAX * 2 - 1, 1);
    }

    void setUniform(unsigned int shaderProgram) {
        char buffer[100];
        int location = -1;

        sprintf(buffer, "ellipsoidCount");
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1i(location, ellipsoids.size()); else printf("uniform %s cannot be set\n", buffer);

        sprintf(buffer, "triangleCount");
        location = glGetUniformLocation(shaderProgram, buffer);
        if (location >= 0) glUniform1i(location, triangles.size()); else printf("uniform %s cannot be set\n", buffer);

        camera->setUniform(shaderProgram);
        light->setUniform(shaderProgram);

        for (int i = 0; i < ellipsoids.size(); i++) {
            ellipsoids[i]->setUniform(shaderProgram, i);
        }
        for (int i = 0; i < triangles.size(); i++) {
            triangles[i]->setUniform(shaderProgram, i);
        }
        for (int i = 0; i < materials.size(); i++) {
            materials[i]->setUniform(shaderProgram, i);
        }
    }
};

class Background {
    GLuint vao, vbo[2];
public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(2, vbo);

        float vertices[] = { -1, -1, -1, 1, 1, 1, 1, -1 };
        float uvs[] = { 0, 0, 0, 1, 1, 1, 1, 0 };

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void draw() {
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

GPUProgram gpuProgram;
Background background;
Scene scene;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    gpuProgram.Create(vertexSource, fragmentSource, "outColor");
    scene.build();
    background.create();
    glutPostRedisplay();
}

void onDisplay() {
    scene.setUniform(gpuProgram.getId());
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    background.draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
    case 'a':
        if (scene.sides >= 25) {
            printf("Max sides (25) reached\n");
            break;
        }
        scene.sides++;
        scene.buildTriangles();
        break;
    case 'g':
        scene.material = 0;
        scene.buildTriangles();
        break;
    case 's':
        scene.material = 1;
        scene.buildTriangles();
        break;
    default:
        break;
    }
    glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}


void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {

}

static float lastSec = 0;
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float sec = time / 1000.0f;
    const float dT = 0.01;
    float timeChanged = sec - lastSec;
    for (float t = 0; t < timeChanged; t += dT) {
        float dt = fmin(dT, timeChanged - t);
        scene.animate(dt);
    }    
    lastSec = sec;
    glutPostRedisplay();
}
