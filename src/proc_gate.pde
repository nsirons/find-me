
PImage bg;

void setup(){
  size(1280, 720, P3D);
  perspective(105 * DEG_TO_RAD, float(width)/float(height), 0.0243, 10000);
  //pixelDensity(displayDensity());
}

int size_angle = 10;
int size_r = 3;
int counter = 0;
int alpha = 0;
int beta = 0;
int r = 2;
int gate_width_px = 350;
void draw() {

  
  String filename = "img" + nf(counter) + "_alpha" + nf(alpha) + "_beta" + nf(beta) + "_R" + nf(r) + ".png"; 
  //float x = random(0.1 * width, 0.9 * width);
  //float y = random(0.1 * height, 0.9 * height);
  bg = loadImage("/home/kit/projects/find-me/data/backgrounds/" + nf(0) + "_background.png");
  filter(POSTERIZE, 4);
  bg.resize(width, height);
  background(bg);
  lights();
  //float rr = 1.4*1280.0/2.0/ (tan(DEG_TO_RAD*105.0/2.0)*r);
  float rr = (1280.0/2.0) / 1.4 * r;
  float eyeX = rr*cos(alpha * DEG_TO_RAD);
  //float eyeY = r*sin(beta * DEG_TO_RAD)*350/1.4;
  //float eyeX = 0;
  //float eyeY = height /2.0 *sin(beta * DEG_TO_RAD);
  float eyeY = rr*sin(beta * DEG_TO_RAD);
  //float eyeZ = (height/2.0) / tan(PI*30.0 / 180.0
  float eyeZ = -0;
  //print(r);
  println(eyeZ, r ,rr);
  float centerX = 0;
  float centerY = 0;
  float centerZ = 0;
  float upX = 0;
  float upY = 1;
  float upZ = 0;
  
  camera(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ);
  //float = 
  pushMatrix();
    translate(0,0, -350);
    //rotateY(alpha * DEG_TO_RAD);
    //rotateX(beta * DEG_TO_RAD);
    //rotateY(angle);
    //fill(200,   100, 100);
    //noStroke();
    stroke(0,0,0);
    strokeWeight(5);
    noFill();
    box(gate_width_px, gate_width_px, 10);
    //rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, 10);
    //rect( gate_width_px/2, -gate_width_px /2, 10, gate_width_px);
    //rect(-gate_width_px/2,  gate_width_px /2, gate_width_px, 10);
    //rect(-gate_width_px/2, -gate_width_px /2, 10, gate_width_px);
  popMatrix();
  pushMatrix();
  //-r*350/1.4
  translate(0,0, -350);
    //translate(0, height/2, 0);
    //rotateY(alpha * DEG_TO_RAD);
    //rotateX(beta * DEG_TO_RAD);
    //rotateY(angle);
    //fill(200,   100, 100);
    //noStroke();
    stroke(255,255,255);
    strokeWeight(5);
    noFill();
    box(gate_width_px-10, gate_width_px-10, 10);
    //rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, 10);
    //rect( gate_width_px/2, -gate_width_px /2, 10, gate_width_px);
    //rect(-gate_width_px/2,  gate_width_px /2, gate_width_px, 10);
    //rect(-gate_width_px/2, -gate_width_px /2, 10, gate_width_px);
  popMatrix();
  pushMatrix();
  translate(0,0, -350);
    //translate(width/2, height/2, 0);
    //rotateY(alpha * DEG_TO_RAD);
    //rotateX(beta * DEG_TO_RAD);
    //rotateY(angle);
    //fill(200,   100, 100);
    //noStroke();
    stroke(255,255,255);
    strokeWeight(5);
    noFill();
    box(gate_width_px+10, gate_width_px+15, 10);
    //rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, 10);
    //rect( gate_width_px/2, -gate_width_px /2, 10, gate_width_px);
    //rect(-gate_width_px/2,  gate_width_px /2, gate_width_px, 10);
    //rect(-gate_width_px/2, -gate_width_px /2, 10, gate_width_px);
    //line(0, 0, 0, 100, 0, 0);
    //line(0, 0, 0, 0, 100, 0);
    //line(0, 0, 0, 0, 0, 100);
  popMatrix();
  
  pushMatrix();
    
    stroke(255,0,0);
    strokeWeight(4);
    line(0, 0, 0, 100, 0, 0);
    
  popMatrix();
    pushMatrix();
    
    stroke(0,0,255);
    strokeWeight(4);
    line(0, 0, 0, 0, 0, 100);
    
  popMatrix();
    pushMatrix();
    
    stroke(0,255,0);
    strokeWeight(4);
    line(0, 0, 0, 0, 100, 0);
    
  popMatrix();
    
  
  saveFrame("../data/ale_inputs/" + filename);
  //if (counter == (size_angle+1)*(size_angle+1)*size_r) noLoop();
    counter += 1;
  if (alpha < 90) {
    alpha += 90 / size_angle;
  }
  else {
      if (alpha == 90 & beta == 90) {
        r += 1;
        beta = 0;
        alpha = 0;
     }
      else{
        alpha = 0;
        beta += 90 / size_angle;
    }
  }
  if (counter == (size_angle+1)*(size_angle+1)*size_r) noLoop();
  //background(255);
  //noLights();
  
  //pushMatrix();
  //  translate(x, y, -depth);
  //  rotateY(angle);
  //  fill(0);
  //  noStroke();
  //  rect(-60, -60, 120, 10);
  //  rect(50, -50, 10, 100);
  //  rect(-60, 50, 120, 10);
  //  rect(-60, -50, 10, 100);
  //popMatrix();
  
  //if (i < steps) saveFrame("../data/ale_outputs/" + filename);
  
  //i++;
  
}
