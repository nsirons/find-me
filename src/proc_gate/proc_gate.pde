
PImage bg;

void setup(){
  size(512, 288, P3D);
  perspective(105 * DEG_TO_RAD, float(width)/float(height), 0.0243, 10000);
}

int counter = 0;
int limit = 7000;
int n_backgrounds = 19;
float angle_min = -0.4 * PI;
float angle_max = 0.4 * PI;
float depth_min = 50;
float depth_max = 120;
float gate_width_px = 300;
float gate_thickness = 10;

void draw() {
  float x = random(0.1 * width, 0.9 * width);
  float y = random(0.1 * height, 0.9 * height);
  float depth = random(depth_min, depth_max); 
  float alpha = random(angle_min, angle_max);
  float beta = random(angle_min, angle_max);
  int filter_mode = (int) random(5) -1;
  int bg_id = (int) random(n_backgrounds+1)-1;

  String filename = "img" + nf(counter) + "_alpha" + nf(alpha) + "_beta" + nf(beta) + "_R" + nf(depth) + ".jpg"; 

  bg = loadImage("../../data/backgrounds/" + nf(bg_id) + "_background.png"); // change path
  filter_mode = 0;
  bg.resize(width, height);
  background(bg);
  lights();
  //println(filter_mode);


  
  //float rr = (1280.0/2.0) / 1.4 * r;
  //float eyeX = 0*cos(alpha * DEG_TO_RAD);
  //float eyeY = 0*sin(beta * DEG_TO_RAD);
  //float eyeZ = 0;
  
  //float centerX = 0;
  //float centerY = 0;
  //float centerZ = 0;
  //float upX = 0;
  //float upY = 1;
  //float upZ = 0;
  
  //camera(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ);

  // black middle part
  //pushMatrix();
  //  translate(x,y, -depth);
  //  rotateY(alpha);
  //  rotateX(beta);
  //  fill(0,0,0);
  //  noStroke();
  //  rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, gate_thickness);
  //  rect( gate_width_px/2, -gate_width_px /2, gate_thickness, gate_width_px);
  //  rect(-gate_width_px/2,  gate_width_px /2, gate_width_px, gate_thickness);
  //  rect(-gate_width_px/2, -gate_width_px /2, gate_thickness, gate_width_px);
    
  //  pushMatrix();
  //    translate(0,0 -gate_thickness);
  //    fill(0,0,0);
  //    noStroke();
  //    rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, gate_thickness);
  //    rect( gate_width_px/2, -gate_width_px /2, gate_thickness, gate_width_px);
  //    rect(-gate_width_px/2,  gate_width_px /2, gate_width_px, gate_thickness);
  //    rect(-gate_width_px/2, -gate_width_px /2, gate_thickness, gate_width_px);
  //  popMatrix();
  //popMatrix();
  
  // white inside part
  for (int layer = 0; layer<1; layer++) {
    
    depth = depth + layer*gate_thickness;
    
    pushMatrix();
      translate(x,y, -depth);
      rotateY(alpha);
      rotateX(beta);
      fill(255,255,255);
      noStroke();
      rect(-gate_width_px/2 + gate_thickness, -gate_width_px /2 + gate_thickness, gate_width_px - 2*gate_thickness, gate_thickness);
      for (int i=0; i<3; i++){
        rotateZ(PI/2);
        rect(-gate_width_px/2 + gate_thickness, -gate_width_px /2 + gate_thickness, gate_width_px - 2*gate_thickness, gate_thickness);
      }
     popMatrix();
      
      // black middle part
      pushMatrix();
      translate(x,y, -depth);
      rotateY(alpha);
      rotateX(beta);
      fill(0,0,0);
      noStroke();
      rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, gate_thickness);
      for (int i=0; i<3; i++){
        rotateZ(PI/2);
        rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, gate_thickness);
      }
  
    popMatrix();
    
      // white inside part
    pushMatrix();
      translate(x,y, -depth);
      rotateY(alpha);
      rotateX(beta);
      fill(255,255,255);
      noStroke();
      rect(-gate_width_px/2 - gate_thickness, -gate_width_px /2 - gate_thickness, gate_width_px + 2*gate_thickness, gate_thickness);
      for (int i=0; i<3; i++){
        rotateZ(PI/2);
        rect(-gate_width_px/2 - gate_thickness, -gate_width_px /2 - gate_thickness, gate_width_px + 2*gate_thickness, gate_thickness);
      }
     popMatrix();
 

}
  

  
  
   
    // white outside part
  //pushMatrix();
  //  rotateY(alpha);
  //  rotateX(beta);
  //  fill(255,255,255);
  //  noStroke();
  //  for (int layer=0; layer<=1; layer++) {
  //    switch (layer){
  //      case 0:
  //        translate(x,y, -depth);
  //      case 1:
  //        translate(0,0, -gate_thickness);
  //      }

  //    rect(-gate_width_px/2 - gate_thickness, -gate_width_px /2 - gate_thickness, gate_width_px + 2*gate_thickness, gate_thickness);
  //        fill(255,255,255);
  //  noStroke();
  //    for (int i=0; i<3; i++){
  //      rotateZ(PI/2);
  //      fill(255,255,255);
  //      noStroke();
  //      rect(-gate_width_px/2 - gate_thickness, -gate_width_px /2 - gate_thickness, gate_width_px + 2*gate_thickness, gate_thickness);
  //    }
  //  }
  // popMatrix();
  
  //pushMatrix();
  //    translate(x,y, -depth + -gate_thickness);
  //    rotateY(alpha);
  //    rotateX(beta);
  //    fill(0,0,0);
  //    noStroke();
  //    rect(-gate_width_px/2, -gate_width_px /2, gate_width_px, gate_thickness);
  //    rect( gate_width_px/2, -gate_width_px /2, gate_thickness, gate_width_px);
  //    rect(-gate_width_px/2,  gate_width_px /2, gate_width_px, gate_thickness);
  //    rect(-gate_width_px/2, -gate_width_px /2, gate_thickness, gate_width_px);
  // popMatrix();
  
    //  switch(filter_mode) {
    //  case 0:
    //    filter(POSTERIZE,(int)random(2,10));
    //    break;
    //  case 1:
    //    filter(BLUR,(int)random(10,10));
    //    break;
    //  case 2:
    //    filter(ERODE);
    //    break;
    //  case 3:
    //    filter(DILATE);
    //    break;
    //}
  
  saveFrame("../../data/cnn_inputs/" + filename);
  
  background(255);
  noLights();
  
  for (int layer = 0; layer<1; layer++) {
  
  depth = depth + layer*gate_thickness;
  
    pushMatrix();
      translate(x,y, -depth);
      rotateY(alpha);
      rotateX(beta);
      fill(0,0,0);
      noStroke();
      rect(-gate_width_px/2 - gate_thickness, -gate_width_px /2 - gate_thickness, gate_width_px + 2*gate_thickness, 3*gate_thickness);
      for (int i=0; i<3; i++){
        rotateZ(PI/2);
        rect(-gate_width_px/2 - gate_thickness, -gate_width_px /2 - gate_thickness, gate_width_px + 2*gate_thickness, 3*gate_thickness);
      }
   popMatrix();
  }
  

  
  saveFrame("../../data/cnn_targets/" + filename);
  if (counter == limit) noLoop();
  counter ++;
}
