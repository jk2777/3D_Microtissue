// Gmsh project created on Wed Nov  6 15:07:48 2019
resolution2 = 0.04;
resolution1 = 0.04;

///////////////////// Points ///////////////////////
// Outer surface
Point(0)   = { 0.00,  0.00, 0.00, resolution1};
Point(1)   = {-0.75, -0.31, 0.00, resolution1};
Point(2)   = {-0.75,  0.00, 0.00, resolution1};
Point(3)   = {-0.75,  0.31, 0.00, resolution1};
Point(4)   = {-0.50,  0.56, 0.00, resolution1};
Point(5)   = { 0.00,  0.56, 0.00, resolution1};
Point(6)   = { 0.50,  0.56, 0.00, resolution1};
Point(7)   = { 0.75,  0.31, 0.00, resolution1};
Point(8)   = { 0.75,  0.00, 0.00, resolution1};
Point(9)   = { 0.75, -0.31, 0.00, resolution1};
Point(10)  = { 0.50, -0.56, 0.00, resolution1};
Point(11)  = { 0.00, -0.56, 0.00, resolution1};
Point(12)  = {-0.50, -0.56, 0.00, resolution1};

// Cylinder 1
Point(21) = { 0.50, 0.31, 0.00, resolution1};
Point(22) = { 0.55, 0.31, 0.00, resolution2};
Point(23) = { 0.50, 0.36, 0.00, resolution2};
Point(24) = { 0.45, 0.31, 0.00, resolution2};
Point(25) = { 0.50, 0.26, 0.00, resolution2};

// Cylinder 2
Point(31) = {-0.50, 0.31, 0.00, resolution1};
Point(32) = {-0.45, 0.31, 0.00, resolution2};
Point(33) = {-0.50, 0.36, 0.00, resolution2};
Point(34) = {-0.55, 0.31, 0.00, resolution2};
Point(35) = {-0.50, 0.26, 0.00, resolution2};

// Cylinder 3
Point(41) = {-0.50, -0.31, 0.00, resolution1};
Point(42) = {-0.45, -0.31, 0.00, resolution2};
Point(43) = {-0.50, -0.26, 0.00, resolution2};
Point(44) = {-0.55, -0.31, 0.00, resolution2};
Point(45) = {-0.50, -0.36, 0.00, resolution2};

// Cylinder 4
Point(51) = { 0.50, -0.31, 0.00, resolution1};
Point(52) = { 0.55, -0.31, 0.00, resolution2};
Point(53) = { 0.50, -0.26, 0.00, resolution2};
Point(54) = { 0.45, -0.31, 0.00, resolution2};
Point(55) = { 0.50, -0.36, 0.00, resolution2};

//////////////////////// Line ///////////////////////
// Shapes
Line(1) = {1, 2};
Line(2) = {2, 3};
Circle(3) = {3, 31, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Circle(6) = {6, 21, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Circle(9) = {9, 51, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Circle(12) = {12, 41, 1};
Curve Loop(13) = {1,2,3,4,5,6,7,8,9,10,11,12};
Line(14) = {0, 2};
Line(15) = {0, 5};
Line(16) = {0, 8};
Line(17) = {0, 11};

// Cylinders
Circle(21) = {22, 21, 23};
Circle(22) = {23, 21, 24};
Circle(23) = {24, 21, 25};
Circle(24) = {25, 21, 22};
Curve Loop(25) = {21,22,23,24};

Circle(31) = {32, 31, 33};
Circle(32) = {33, 31, 34};
Circle(33) = {34, 31, 35};
Circle(34) = {35, 31, 32};
Curve Loop(35) = {31,32,33,34};

Circle(41) = {42, 41, 43};
Circle(42) = {43, 41, 44};
Circle(43) = {44, 41, 45};
Circle(44) = {45, 41, 42};
Curve Loop(45) = {41,42,43,44};

Circle(51) = {52, 51, 53};
Circle(52) = {53, 51, 54};
Circle(53) = {54, 51, 55};
Circle(54) = {55, 51, 52};
Curve Loop(55) = {51,52,53,54};


////////////////////// Surface ///////////////////////
Curve Loop(56) = {6, 7, -16, 15, 5};
Plane Surface(1) = {25, 56};
Curve Loop(57) = {14, 2, 3, 4, -15};
Plane Surface(2) = {35, 57};
Curve Loop(58) = {17, 11, 12, 1, -14};
Plane Surface(3) = {45, 58};
Curve Loop(59) = {8, 9, 10, -17, 16};
Plane Surface(4) = {55, 59};

////////////////////// EXTRUDE ///////////////////////
Extrude {0, 0, 0.11/2} {
  Surface{4}; Surface{1}; Surface{2}; Surface{3}; 
}
Extrude {0, 0, -0.11/2} {
  Surface{3}; Surface{2}; Surface{1}; Surface{4}; 
}

/////////////////// PHYSICAL DOMAIN ///////////////////
Physical Volume(99) = {5, 4, 3, 6, 2, 7, 8, 1};
//Physical Surface(1) = {367, 355, 359, 363, 132, 120, 124, 128, 320, 308, 312, 316, 179, 167, 171, 175, 273, 261, 226, 269, 222, 265, 218, 214, 414, 402, 85, 406, 410, 81, 77, 73};
Physical Surface(1) = {120, 124, 128, 132, 359, 355, 367, 363};
Physical Surface(2) = {171, 167, 179, 312, 175, 316, 308, 320};
Physical Surface(3) = {218, 214, 226, 222, 265, 261, 273, 269};
Physical Surface(4) = {73, 77, 81, 85, 402, 406, 414, 410};
