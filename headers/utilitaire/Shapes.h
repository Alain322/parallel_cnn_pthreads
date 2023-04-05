#ifndef SHAPES_H
#define SHAPES_H

typedef struct Shapes
{
    int depth;
    int height;
    int width;
}Shapes;


Shapes new_Shapes(int depth , int height , int width);
void free_Shapes(Shapes Shapes);


#endif