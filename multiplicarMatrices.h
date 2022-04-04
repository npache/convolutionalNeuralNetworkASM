
typedef struct str_matrix {
 double** matrix;
 uint16_t height;
 uint16_t width; 
} Matrix_t;


typedef struct str_image {
 Matrix_t* image;
 uint16_t height;
 uint16_t width;
 uint16_t channels; 
} Image_t;


typedef struct str_filter{
 Matrix_t** filter;
 uint16_t f;
 uint16_t depth_in;
 uint16_t depth_out; 

} Filter_t;
