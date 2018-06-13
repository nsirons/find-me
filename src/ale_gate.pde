
PImage bg;

void setup(){
  size(365, 240,P3D);
  //pixelDensity(displayDensity());
}

float angle_min = -0.4 * PI;
float angle_max = 0.4 * PI;
float depth_min = 100;
float depth_max = 500;
int steps = 10000;
int i = 0;

void draw() {
  float angle = random(angle_min, angle_max);
  float depth = random(depth_min, depth_max);
  String filename = nf(i) + "_a" + nf(angle) + "d" + nf(depth) + ".png"; 
  float x = random(0.1 * width, 0.9 * width);
  float y = random(0.1 * height, 0.9 * height);
  bg = loadImage("../data/backgrounds/" + nf(i % 20) + "_background.png");
  filter(POSTERIZE, 4);
  bg.resize(width, height);
  background(bg);
  lights();
  
  pushMatrix();
    translate(x, y, -depth);
    rotateY(angle);
    fill(200, 100, 100);
    noStroke();
    rect(-60, -60, 120, 10);
    rect(50, -50, 10, 100);
    rect(-60, 50, 120, 10);
    rect(-60, -50, 10, 100);
  popMatrix();
  
  if (i < steps) saveFrame("../data/ale_inputs/" + filename);
  
  background(255);
  noLights();
  
  pushMatrix();
    translate(x, y, -depth);
    rotateY(angle);
    fill(0);
    noStroke();
    rect(-60, -60, 120, 10);
    rect(50, -50, 10, 100);
    rect(-60, 50, 120, 10);
    rect(-60, -50, 10, 100);
  popMatrix();
  
  if (i < steps) saveFrame("../data/ale_outputs/" + filename);
  
  i++;
}
