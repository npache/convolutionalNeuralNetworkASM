//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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include "../tp2.h"
#include "tiempo.h"
#include "libbmp.c"
#include "utils.c"
#include "imagenes.c"
#include "../cli.c"
#include "multiplicarMatrices.h"

void print_filter(char* s, Filter_t filt){
FILE * fp;
fp = fopen (s, "w+");
for(int i=0;i<filt.depth_out;i++){
      for(int j=0;j<filt.depth_in;j++){
        for(int k=0;k<filt.f;k++){
          for(int l=0;l<filt.f;l++){
    fprintf(fp,"%.10f \n",filt.filter[i][j].matrix[k][l]);
    }
  } 
}
}
fclose(fp);
}

void print_image(char* s, Image_t m){
FILE * fp;
fp = fopen (s, "w+");
for(int i = 0;i< m.channels;i++){
for(int j = 0;j< m.height;j++){
  for(int k = 0;k< m.width;k++){
    fprintf(fp,"%.10f \n",m.image[i].matrix[j][k]);
    }
  } 
}
fclose(fp);
}


void print_matrix(char* s, Matrix_t m){
FILE * fp;
fp = fopen (s, "w+");
for(int i = 0;i< m.height;i++){
  for(int j = 0;j< m.width;j++){
    fprintf(fp,"%.10f \n",m.matrix[i][j]);
    }
  } 
fclose(fp);
}

void copy_filter(Filter_t tot_f1_copy,Filter_t tot_f1){
  for(int i=0;i<tot_f1_copy.depth_out;i++){
    for(int j=0;j<tot_f1_copy.depth_in;j++){
      for(int k=0;k<tot_f1_copy.f;k++){
        for(int l=0;l<tot_f1_copy.f;l++){
        tot_f1_copy.filter[i][j].matrix[k][l]=tot_f1.filter[i][j].matrix[k][l];

        } 
      }
    }
  }

}

void copy_matrix(Matrix_t tot_f1_copy,Matrix_t tot_f1){
  for(int i=0;i<tot_f1_copy.height;i++){
    for(int j=0;j<tot_f1_copy.width;j++){
 
        tot_f1_copy.matrix[i][j]=tot_f1.matrix[i][j];

        } 
      }
    }

void matrix_initialize_n(Matrix_t* matrix, uint16_t height, uint16_t width){
  matrix->height=height;
  matrix->width=width;
  matrix->matrix = (double **)malloc(height*sizeof(double*));
  
  for(int i = 0;i< height;i++){
      
    matrix->matrix[i]= (double *)malloc( width*sizeof(double));
    for(int j = 0;j< width;j++){
      matrix->matrix[i][j]=0;
      
    }
  }
      
}

void matrix_initialize_n_free(Matrix_t matrix){
  
  for(int i = 0;i< matrix.height;i++){
    free(matrix.matrix[i]);
  }
   free(matrix.matrix);
}
void matrix_mul_scalar(Matrix_t matrix, double x){
  for(int i=0;i<matrix.height;i++){
    for(int j=0;j<matrix.width;j++){
      matrix.matrix[i][j]=matrix.matrix[i][j]*x;
    }
  }
}

void matrix_add_scalar(Matrix_t matrix, double x){
  for(int i=0;i<matrix.height;i++){
    for(int j=0;j<matrix.width;j++){
      matrix.matrix[i][j]=matrix.matrix[i][j]+x;
    }
  }
}

void matrix_sub_scalar(Matrix_t matrix, double x){
  for(int i=0;i<matrix.height;i++){
    for(int j=0;j<matrix.width;j++){
      matrix.matrix[i][j]=matrix.matrix[i][j]-x;
    }
  }
}
void matrix_div_scalar(Matrix_t matrix, double x){
  for(int i=0;i<matrix.height;i++){
    for(int j=0;j<matrix.width;j++){
      matrix.matrix[i][j]=x/matrix.matrix[i][j];
    }
  }
}

void matrix_sqrt(Matrix_t matrix,int debugger){
  for(int i=0;i<matrix.height;i++){
    for(int j=0;j<matrix.width;j++){
      matrix.matrix[i][j]=sqrt(matrix.matrix[i][j]);
    }
  }
}
     


void dot_n(Matrix_t matrix_A,Matrix_t matrix_B,Matrix_t res)
{
    assert(matrix_A.width==matrix_B.height && "The dimension of the matrices do not match");

    for(int i = 0;i< matrix_A.height;i++){
        for(int j = 0;j< matrix_B.width;j++){
            double sum = 0;
            for(int k = 0;k< matrix_A.width;k++){

                sum+= matrix_A.matrix[i][k]*matrix_B.matrix[k][j];
            }   
            res.matrix[i][j]=sum;  
            
                
        }
    }

}



void matrices_sub(
    double **matrix_A,
    double **matrix_B,
    int width_A,
    int height_A,
    int width_B,
    int height_B){
    assert(width_A==width_B && "The dimension of the matrices do not match");
    assert(height_A==height_B && "The dimension of the matrices do not match");
    for(int i = 0;i< height_A;i++){
        for(int j = 0;j< width_A;j++){
          matrix_A[i][j] = matrix_A[i][j]-matrix_B[i][j];
        }
    }

   
}



void matrices_add_n(Matrix_t matrix_A,Matrix_t matrix_B){
    assert(matrix_A.width==matrix_B.width && "The dimension of the matrices do not match");
    assert(matrix_A.height==matrix_B.height && "The dimension of the matrices do not match");
    for(int i = 0;i< matrix_A.height;i++){
        for(int j = 0;j< matrix_B.width;j++){
          matrix_A.matrix[i][j] = matrix_A.matrix[i][j]+matrix_B.matrix[i][j];
        }
    }

   
}

void transpose_n(Matrix_t matrix_A,Matrix_t transposed){
  for(int i = 0;i< matrix_A.height;i++){
        for(int j = 0;j< matrix_A.width;j++){
          transposed.matrix[j][i] = matrix_A.matrix[i][j];
          //printf("%f \n",transposed.matrix[j][i] );
        }
    }
  }



void matrices_mul_n(
    Matrix_t matrix_A,
    Matrix_t matrix_B){
    assert(matrix_A.width==matrix_B.width && "The dimension of the matrices do not match");
    assert(matrix_A.height==matrix_B.height && "The dimension of the matrices do not match");
    for(int i = 0;i< matrix_A.height;i++){
        for(int j = 0;j< matrix_A.width;j++){
          matrix_A.matrix[i][j] = matrix_A.matrix[i][j]*matrix_B.matrix[i][j];
        }
    }


}
void matrices_sub_n(
    Matrix_t matrix_A,
    Matrix_t matrix_B){
    assert(matrix_A.width==matrix_B.width && "The dimension of the matrices do not match");
    assert(matrix_A.height==matrix_B.height && "The dimension of the matrices do not match");
    for(int i = 0;i< matrix_A.height;i++){
        for(int j = 0;j< matrix_A.width;j++){
          matrix_A.matrix[i][j] = matrix_A.matrix[i][j]-matrix_B.matrix[i][j];
        }
    }


}
double matrices_add_all_elements_n(Matrix_t matrix_A){
    double res = 0;
    for(int i = 0;i< matrix_A.height;i++){
        for(int j = 0;j< matrix_A.width;j++){
         res+=matrix_A.matrix[i][j];
        }
    } 
    return res;
}



void matrices_flatten(
  Image_t matrix, 
  Matrix_t res){
  int counter = 0;
  for(int k = 0;k< matrix.channels;k++){
    for(int i = 0;i< matrix.height;i++){
          for(int j = 0;j< matrix.width;j++){
          res.matrix[counter][0]  = matrix.image[k].matrix[i][j];
          counter++;
          }
      }
  }
 

}



void matrices_reshape_from_flatten_n( 
  Matrix_t matrix, 
  Image_t res){
    int f = 0;
    int g = 0;
    for(int k = 0;k< res.channels;k++){
      for(int i = 0;i< res.height;i++){
        for(int j = 0;j< res.width;j++){
          res.image[k].matrix[i][j]=matrix.matrix[f][g];
          g++;
          if(g>=matrix.width){
             g=0;
             f++; 
          }
        }
      }
    }
  }
void initializeFilter(Filter_t* filt, uint16_t depth_out, uint16_t depth_in, uint16_t f){
  filt->f=f;
  filt->depth_in=depth_in;
  filt->depth_out=depth_out;
  filt->filter = (Matrix_t**)malloc(depth_out*sizeof(Matrix_t*));
  
  for(int i = 0;i<depth_out;i++){
    filt->filter[i] = (Matrix_t*)malloc(depth_out*sizeof(Matrix_t));
    for(int j = 0;j<depth_in;j++){  
      Matrix_t temp;
      matrix_initialize_n(&temp,f,f);
      filt->filter[i][j]=temp;
    } 
  }
}

void image_initialize(
  Image_t* image,
  unsigned int depth,
  unsigned int height,
  unsigned int width){ 
  image->channels=depth;
  image->height=height;
  image->width=width;
  for(int i = 0;i< depth;i++){
      Matrix_t temp;
      
      matrix_initialize_n(&temp,height,width);
      
      image->image[i]=temp;
      
    }

  }
  
void image_initialize_free(
  Image_t image){
  for(int i = 0;i< image.channels;i++){
     matrix_initialize_n_free(image.image[i]);
    }
  }
void initializeFilterFree(Filter_t filt){
  
  
  for(int i = 0;i<filt.depth_out;i++){
    for(int j = 0;j<filt.depth_in;j++){  
      matrix_initialize_n_free(filt.filter[i][j]);
    } 
  }
}

 
void image_to_matrix(//RGB
  bgra_t* src_matrix,
  Image_t* res,
  uint16_t depth, uint16_t height,uint16_t width){
  int c =0;
  res->channels=depth;
  res->height=height;
  res->width=width;
  double* pixels = (double *)malloc(res->channels*res->width*res->height*sizeof(double));
  for(int i  = 0; i<150;i++){
      for(int j  = 0; j<150;j++){
        pixels[c]=src_matrix[i*150+ j].r;
        pixels[c+1]=src_matrix[i*150+ j].g;
        pixels[c+2]=src_matrix[i*150+ j].b;
        c+=3;
        
      }
      
  }


  c=0;
      
  res->image = (Matrix_t*)malloc(res->channels*sizeof(Matrix_t));
  for(int i = 0;i<res->channels;i++){
    Matrix_t temp;
    matrix_initialize_n(&temp,res->height,res->width);
    res->image[i]=temp;
    for (int j = 0; j < res->height; j++){
      for (int k = 0; k < res->width; k++){
        res->image[i].matrix[j][k]=pixels[c];
        //printf("%f\n", pixels[c]);
        c++;
      }
    }
  }
  free(pixels);

}

void slice_n(
  Image_t matrix_A,
  int start_y,
  int start_x,
  int step,
  Image_t res){

   assert(start_x+step<=matrix_A.width && "Limit out of bounds");
   assert(start_y+step<=matrix_A.height && "Limit out of bounds");

   for(int t =0;t<matrix_A.channels;t++){ 
     int k = 0;
     for(int i = start_y;i<start_y+step;i++){
        int u=0;
        for(int j = start_x;j<start_x+step;j++){
            
            res.image[t].matrix[k][u] = matrix_A.image[t].matrix[i][j]; 
            
            u++;
            
        }    
        k++;
             
     }
   }
   

}






void slice_free_n(
  int start_y,
  int step,
  Image_t res){
  for(int t =0;t<res.channels;t++){ 
    int k=0;
    for(int p = start_y;p<start_y+step;p++){
      free(res.image[t].matrix[k]); 
      k++;
    }
    free(res.image[t].matrix);   
  }

}

void convolution(Image_t image,Filter_t filt,Matrix_t bias,int s, int f, Image_t res,int debugger){
   assert(image.channels==filt.depth_in && "Dimensions of filter must match dimensions of input image");
   
   FILE * fp;

   fp = fopen ("conv.txt", "w+");
 
   
   for(int i = 0;i< filt.depth_out;i++){
    int curr_y = 0;
    int out_y = 0;
    
    while(curr_y + f <= image.height){
      int curr_x=0;
      int out_x = 0;
      while(curr_x + f <= image.height){
          double sum = 0;
         
          Image_t slice_image;   
          slice_image.image = (Matrix_t*)malloc(image.channels*sizeof(Matrix_t));
          
          image_initialize(&slice_image,image.channels,f,f);
          slice_n(image,curr_y,curr_x,f,slice_image);
          for(int j =0;j<image.channels;j++){
            matrices_mul_n(slice_image.image[j],filt.filter[i][j]);
            sum+= matrices_add_all_elements_n(slice_image.image[j]);
                      }
          image_initialize_free(slice_image);
          
          free(slice_image.image);
        
          sum+=bias.matrix[i][0];
        
       
        fprintf(fp, "%d\n",(int)sum);
        res.image[i].matrix[out_y][out_x] =sum;
        curr_x+=s;
        out_x+=1;
      }
      curr_y+=s;
      out_y+=1;
    }
   }


   fclose(fp);

}




void maxpool(Image_t image,unsigned int f, unsigned int s, Image_t res){
    ///FILE * fp;

    //fp = fopen ("d.txt", "w+");
    for(unsigned int i = 0;i<image.channels;i++){
        int curr_y=0;
        int out_y=0;
        while(curr_y+f<=image.height){
            int curr_x =0;
            int out_x  =0;
            while(curr_x+f<=image.width){
                Image_t slice_image;   
                slice_image.image = (Matrix_t*)malloc(image.channels*sizeof(Matrix_t));
                
                image_initialize(&slice_image,image.channels,f,f);
                slice_n(image,curr_y,curr_x,f,slice_image);
                
                double max=slice_image.image[i].matrix[0][0];
                for(int j=0;j<f;j++){
                    for(int k=0;k<f;k++){
                        if(max<slice_image.image[i].matrix[j][k])max=slice_image.image[i].matrix[j][k];
                    }  
                }
                res.image[i].matrix[out_y][out_x]=max;
                //fprintf(fp, "%d\n",(int)max);
                curr_x+=s;
                out_x+=1;

             
                image_initialize_free(slice_image);
                free(slice_image.image);
        

            }
            curr_y+=s;
            out_y+=1;

      
        }

    }
    //
  
  
  //fclose(fp);  
}


void backwardConvolution(Image_t conv_in,Image_t dconv_prev,Filter_t dfilt,Filter_t filt,Matrix_t dbias,int s, int f,Image_t dout){
      assert(conv_in.channels==filt.depth_in && "Dimensions of filter must match dimensions of input image");
  
  ////FILE * fp;
    ////fp = fopen ("backConv_.txt", "w+");
   for(int i = 0;i< filt.depth_out;i++){
    int curr_y = 0;
    int out_y = 0;
    
    while(curr_y + f <= conv_in.width){
      int curr_x=0;
      int out_x = 0;
      while(curr_x + f <= conv_in.width){
          
        
        for(int j = 0;j<filt.depth_in;j++){
            for(int k = 0;k<f;k++){
              for(int m = 0;m<f;m++){

                dfilt.filter[i][j].matrix[k][m]+= dconv_prev.image[i].matrix[out_y][out_x]*conv_in.image[j].matrix[curr_y+k][curr_x+m];
              }
            }
          }
        for(int j = 0;j<conv_in.channels;j++){
          for(int k = 0;k<f;k++){
            for(int m = 0;m<f;m++){
              
              //printf("%d %d %d",j,k,m,image_channels);
              dout.image[j].matrix[curr_y+k][curr_x+m]+=(dconv_prev.image[i].matrix[out_y][out_x])*(filt.filter[i][j].matrix[k][m]);
            }
          }
        }
        
       
        curr_x+=s;
        out_x+=1;
      }
      curr_y+=s;
      out_y+=1;
    }
    dbias.matrix[i][0]+=matrices_add_all_elements_n(dconv_prev.image[i]);

    
   }

   //fclose(fp);
  

}



void backwardMaxpool(Image_t dpool,Image_t image,unsigned int f, unsigned int s, Image_t res){
    //FILE * fp;
    //fp = fopen ("backwardMaxpool.txt", "w+");
    for(unsigned int i = 0;i<image.channels;i++){
        int curr_y=0;
        int out_y=0;
        while(curr_y+f<=image.height){
            int curr_x =0;
            int out_x  =0;
            while(curr_x+f<=image.width){
                Image_t slice_image;   
                slice_image.image = (Matrix_t*)malloc(image.channels*sizeof(Matrix_t));
                
                image_initialize(&slice_image,image.channels,f,f);
                slice_n(image,curr_y,curr_x,f,slice_image);
                
                double max=slice_image.image[i].matrix[0][0];
                
                int a=0;
                int b=0;
                for(int j=0;j<f;j++){
                    for(int k=0;k<f;k++){
                        if(max<slice_image.image[i].matrix[j][k]){
                          max=slice_image.image[i].matrix[j][k];
                          a=j;
                          b=k;
                        }
                    }  
                }
                res.image[i].matrix[curr_y+a][curr_x+b]=dpool.image[i].matrix[out_y][out_x];
                              
               // fprintf(fp, "%d\n",(int)dpool[i][out_y][out_x]);
                
             
                image_initialize_free(slice_image);
                free(slice_image.image);
                curr_x+=s;
                out_x+=1;

            }
            curr_y+=s;
            out_y+=1;

      
        }


    }

  //fclose(fp);  
}

void softmax(Matrix_t x,int size){
  
  for(int i=0;i<size;i++){
    x.matrix[i][0]=1/(1+exp(-x.matrix[i][0]));
    
  }
}

double categoricalCrossEntropy(Matrix_t probs,int** label,int size){ 
  double res=0;
  for(int i=0;i<size;i++){
    //printf("%f %f\n",probs.matrix[i][0],log(probs.matrix[i][0]));
    res+=(label[i][0]*log(probs.matrix[i][0]));}
  return -res;


}


int rand_comparison(const void *a, const void *b)
{
    (void)a; (void)b;

    return rand() % 2 ? +1 : -1;
}
void random_batch(int n, int* vektor,int tot_img){
    for(int i=0;i<tot_img;i++)vektor[i]=i;
   
    qsort(vektor, n, sizeof(int), rand_comparison);
    //for(int i=0;i<tot_img;i++)printf("%d\n",vektor[i]);
     
}


void matrices_relu(Image_t image){
  for(int k = 0;k< image.channels;k++){
      for(int i = 0;i< image.height;i++){
          for(int j = 0;j< image.width;j++){
            if (image.image[k].matrix[i][j]<=0)image.image[k].matrix[i][j] =0;
        }
      }
    }
  }
void matrices_drelu(Image_t dimage,Image_t image){
  for(int k = 0;k< dimage.channels;k++){
      for(int i = 0;i< dimage.height;i++){
          for(int j = 0;j< dimage.width;j++){
            if (image.image[k].matrix[i][j]<=0)dimage.image[k].matrix[i][j] =0;
        }
      }
    }
  }
double apply_conv(Image_t image, int** labels,  int classes,int f,int conv_s, Filter_t filt1 ,Matrix_t bias1, int pool_f,int pool_s,Filter_t filt2,Matrix_t bias2,Matrix_t w3,Matrix_t w4,double*** grad_w,double**** grad_f,double**** grad_b,Matrix_t b3,Matrix_t b4,Filter_t dfilt1,Filter_t dfilt2,Matrix_t dbias1,Matrix_t dbias2,Matrix_t dbias3,Matrix_t dbias4, Matrix_t dw3, Matrix_t dw4, int* predicted){
    
    //printf("%f\n",image.image[0].matrix[0][0]);
    /*
    ///double*** image = (double ***)malloc( image_channels*sizeof(double**));

    ///image_to_matrix(src_matrix,image_dim,image_dim,image);
    /*START CONV1*/
    //printf("%f\n",image.image[0].matrix[0][0]);
    Image_t conv1; 
    conv1.image = (Matrix_t*)malloc(filt1.depth_out*sizeof(Matrix_t));
    int conv1_image = (int)((image.height - f)/conv_s)+1;
    image_initialize(&conv1,filt1.depth_out,conv1_image,conv1_image);
    convolution(image,filt1,bias1,conv_s,f,conv1,0);
      
    //printf("%d %d %d\n",conv1.channels,conv1.height,conv1.width );
    matrices_relu(conv1);
    /*END CONV1*/
    /*START CONV2*/
    Image_t conv2; 
    conv2.image = (Matrix_t*)malloc(filt2.depth_out*sizeof(Matrix_t));
    int conv2_image = (int)((conv1.height - f)/conv_s)+1;
    image_initialize(&conv2,filt2.depth_out,conv2_image,conv2_image);

    convolution(conv1,filt2,bias2, conv_s, f,conv2,0);
    ////printf("%f\n",conv2.image[0].matrix[0][0]);
    
    matrices_relu(conv2);
    //printf("%d %d %d\n",conv2.channels,conv2.height,conv2.width );

    /*END CONV2   */ 
    
    
    unsigned int height=(unsigned int)((conv2_image - pool_f)/pool_s)+1;
    unsigned int width=(unsigned int)((conv2_image - pool_f)/pool_s)+1;
    
    Image_t pooled; 
    pooled.image = (Matrix_t*)malloc(filt2.depth_out*sizeof(Matrix_t));

    
    image_initialize(&pooled,filt2.depth_out,height,width);
    maxpool(conv2, pool_f,  pool_s,pooled);
    //printf("%d %d %d\n",pooled.channels,pooled.height,pooled.width );
    ////printf("%f\n",pooled.image[0].matrix[0][0]);
    
    Matrix_t fc;
    matrix_initialize_n(&fc,filt2.depth_out*width*height,1);
   
    matrices_flatten( pooled,fc);
    
    
    Matrix_t z;
    matrix_initialize_n(&z,128,1);
    dot_n(w3,fc,z);
    matrices_add_n(z,b3);  
    //printf("%f\n",z.matrix[0][0]);

    //printf("%f\n",z.matrix[1][0]);

    //printf("%f\n",z.matrix[126][0]);

    //printf("%f\n",z.matrix[127][0]);
    //RELU
    for(int i=0;i<128;i++){
      for(int j=0;j<1;j++){
       if(z.matrix[i][j]<=0)z.matrix[i][j]=0;
          
      }    
    }
    
    Matrix_t last;
    
    matrix_initialize_n(&last,2,1);
    dot_n(w4,z,last);
/*
    printf("%f %f\n",w3.matrix[0][0],w3.matrix[0][1]);
    printf("%f %f\n",b3.matrix[0][0],b3.matrix[1][0]);
    printf("%f %f\n",w4.matrix[0][0],w4.matrix[0][1]);
    printf("%f %f\n",b4.matrix[0][0],b4.matrix[1][0]);
    printf("%s\n","z");
    printf("%f %f\n",z.matrix[0][0],z.matrix[1][0]);
    printf("%f %f\n",last.matrix[0][0],last.matrix[1][0]);
    */
    matrices_add_n(last,b4); 
    //printf("%f %f\n",last.matrix[0][0],last.matrix[1][0]);
    
    
    softmax(last,2);    
    //printf("%f %f \n ",last.matrix[0][0],last.matrix[0][1]);
    if (last.matrix[0][0]>last.matrix[1][0]){
      *predicted=0;
    }  else{
      *predicted=1;
    }

    double loss = categoricalCrossEntropy(last,labels,2);
  /*  ################################################
      ############# Backward Operation ###############
      ################################################ */ 
    Matrix_t dout;
    
    matrix_initialize_n(&dout,classes,1);
    
  
    for(int i=0;i<classes;i++){
      dout.matrix[i][0]=last.matrix[i][0]-(double)labels[i][0];
    }

    Matrix_t z_transpose;
    matrix_initialize_n(&z_transpose,z.width,z.height);
    transpose_n(z,z_transpose);
    dot_n(dout,z_transpose,dw4);
    //TODO faltaria hacer db4 pero parecer ser lo mismo que dout
    
    dbias4.matrix[0][0] =dout.matrix[0][0];
    dbias4.matrix[1][0] =dout.matrix[1][0];
    /*
    printf("%s %f\n","last[0]",last.matrix[0][0]);
    printf("%s %f\n","last[1]",last.matrix[1][0]);


    printf("%s %f\n","labels[0]",(double)labels[0][0]);
    printf("%s %f\n","labels[1]",(double)labels[1][0]);

    printf("%s %f\n","dout[0]",dout.matrix[0][0]);
    printf("%s %f\n","dout[1]",dout.matrix[1][0]);
    
    printf("%s %f\n","z_transpose[0]",z_transpose.matrix[0][0]);
    printf("%s %f\n","z_transpose[1]",z_transpose.matrix[0][1]);
    printf("%s %f\n","z_transpose[2]",z_transpose.matrix[0][2]);
    printf("%s %f\n","z_transpose[3]",z_transpose.matrix[0][3]);
    printf("%s %f\n","z_transpose[4]",z_transpose.matrix[0][4]);
    printf("%s %f\n","z_transpose[5]",z_transpose.matrix[0][5]);
    printf("%s %f\n","z_transpose[6]",z_transpose.matrix[0][6]);
    printf("%s %f\n","z_transpose[7]",z_transpose.matrix[0][7]);
    

    printf("%s %f\n","dw[0]",dw4.matrix[0][4]);
    printf("%s %f\n","dw[1]",dw4.matrix[0][7]);
    printf("%s %f\n","dw[2]",dw4.matrix[0][8]);
    */
    Matrix_t w4_transpose;
    matrix_initialize_n(&w4_transpose,w4.width,w4.height);
        

    transpose_n(w4,w4_transpose);
    
    Matrix_t dz;
    matrix_initialize_n(&dz,z.height,z.width);
         
   

    dot_n(w4_transpose,dout,dz);
    FILE * fp;

    fp = fopen ("dz.txt", "w+");
    for(int i=0;i<dz.height;i++){
      for(int j=0;j<dz.width;j++){
    
      fprintf(fp,"%.40f \n",dz.matrix[i][j]);
      }
     }
     print_matrix("z_n",z);
    for(int i=0;i<dz.height;i++){
      for(int j=0;j<dz.width;j++){
       if((z.matrix[i][j]*100000000)<=0)dz.matrix[i][j]=0;
      }    
    }
    
    fclose(fp);
    Matrix_t fc_transpose;
    matrix_initialize_n(&fc_transpose,fc.width,fc.height);
    

    transpose_n(fc,fc_transpose);



    

    dot_n(dz,fc_transpose,dw3);
    
    copy_matrix(dbias3,dz);
    //TODO faltaria hacer db3 pero parecer ser lo mismo que dz
    Matrix_t w3_transpose;
    matrix_initialize_n(&w3_transpose,w3.width,w3.height);
    

    transpose_n(w3,w3_transpose);
    
    Matrix_t dfc;

    matrix_initialize_n(&dfc,fc.height,fc.width);
    
    
    dot_n(w3_transpose,dz,dfc);
    //print_matrix("w3_transpose_n",w3_transpose);
    //print_matrix("dz_n",dz);
    //print_matrix("dfc_n",dfc);

   
    Image_t dpool; 
    dpool.image = (Matrix_t*)malloc(filt2.depth_out*sizeof(Matrix_t));
    
    
    image_initialize(&dpool,filt2.depth_out,height,width);
    print_matrix("dfc_n_",dfc);
    //printf("%d %d %d\n", filt2.depth_out,height,width);

    //printf("%d %d\n", fc.height,fc.width);
    matrices_reshape_from_flatten_n(dfc,dpool);

    //Hasta aca anda re piola
    
    Image_t dconv2; 
    dconv2.image = (Matrix_t*)malloc(filt2.depth_out*sizeof(Matrix_t));
    
    image_initialize(&dconv2,filt2.depth_out,conv2_image,conv2_image);
    backwardMaxpool(dpool,conv2,pool_f,pool_s,dconv2);
    
    matrices_drelu( dconv2,conv2);
    Image_t dconv1; 
    dconv1.image = (Matrix_t*)malloc(filt1.depth_out*sizeof(Matrix_t));
    image_initialize(&dconv1,filt1.depth_out,conv1_image,conv1_image);
    backwardConvolution(conv1,dconv2,dfilt2,filt2,dbias2,conv_s,f, dconv1);

    
    matrices_drelu( dconv1,conv1);
    Image_t dimage; 
    dimage.image = (Matrix_t*)malloc(filt1.depth_out*sizeof(Matrix_t));
    image_initialize(&dimage,image.channels,image.width,image.width);
    backwardConvolution(image,dconv1,dfilt1,filt1,dbias1,conv_s,f, dimage);
    /*printf("%s %f\n","df[0]",dfilt1.filter[0][0].matrix[0][0]);
    printf("%s %f\n","df[1]",dfilt1.filter[0][0].matrix[0][1]);
    printf("%s %f\n","df[2]",dfilt1.filter[0][0].matrix[0][2]);
    printf("%s %f\n","df[3]",dfilt1.filter[0][0].matrix[0][3]);
    printf("%s %f\n","db[0]",dbias1.matrix[0][0]);
    printf("%s %f\n","db[1]",dbias1.matrix[1][0]);
    printf("%s %f\n","db[2]",dbias1.matrix[2][0]);
    printf("%s %f\n","db[3]",dbias1.matrix[3][0]);*/
    print_filter("f1_n",filt1);
    print_filter("df1_n",dfilt1);
    print_filter("df2_n",dfilt2);
    print_matrix("dbias1_n",dbias1);
    print_matrix("dbias2_n",dbias2);
    print_matrix("dbias3_n",dbias3);
    print_matrix("dbias4_n",dbias4);
    print_matrix("dw3_n",dw3);
    print_matrix("dw4_n",dw4);
    
    
    
    image_initialize_free(conv1);
    image_initialize_free(conv2);
    image_initialize_free(pooled); 
    image_initialize_free(dpool);  
    image_initialize_free(dconv2);  
    image_initialize_free(dconv1);  
    image_initialize_free(dimage);  
    matrix_initialize_n_free(fc);
    matrix_initialize_n_free(z);
    matrix_initialize_n_free(last);
    matrix_initialize_n_free(dout);
    matrix_initialize_n_free(z_transpose);
    matrix_initialize_n_free(w4_transpose);
    matrix_initialize_n_free(dz);
    matrix_initialize_n_free(fc_transpose);
    matrix_initialize_n_free(w3_transpose);
    matrix_initialize_n_free(dfc);
      
    return loss;
}

void adam_filter(Filter_t f,Filter_t s, Filter_t d, double beta1, double beta2, double lr,int debugger){
  Filter_t d_copy;
  initializeFilter(&d_copy,d.depth_out,d.depth_in,d.f);
  copy_filter(d_copy,d);
  for(int i =0;i<d_copy.depth_out;i++){
    for(int j =0;j<d_copy.depth_in;j++){
      matrix_mul_scalar(s.filter[i][j],beta2);
      matrices_mul_n(d_copy.filter[i][j],d_copy.filter[i][j]);
      matrix_mul_scalar(d_copy.filter[i][j],1-beta2);
      matrices_add_n(s.filter[i][j],d_copy.filter[i][j]);
      }
    }
  Filter_t s_copy;
  initializeFilter(&s_copy,s.depth_out,s.depth_in,s.f);
  copy_filter(s_copy,s);
  for(int i =0;i<s.depth_out;i++){
    for(int j =0;j<s.depth_in;j++){
      matrix_add_scalar(s_copy.filter[i][j],0.0000001);  
      matrix_sqrt(s_copy.filter[i][j],debugger);
      matrix_div_scalar(s_copy.filter[i][j],lr);  
      matrices_mul_n(s_copy.filter[i][j],d.filter[i][j]);
      matrices_sub_n(f.filter[i][j],s_copy.filter[i][j]);
      
      }
    }  
  

  }

void adam_matrix(Matrix_t w,Matrix_t s, Matrix_t d, double beta1, double beta2, double lr,int debugger){
  Matrix_t d_copy;
  matrix_initialize_n(&d_copy,d.height,d.width);
  copy_matrix(d_copy,d);
  matrix_mul_scalar(s,beta2);
  matrices_mul_n(d_copy,d_copy);
  matrix_mul_scalar(d_copy,1-beta2);
  matrices_add_n(s,d_copy);
  Matrix_t s_copy;
  matrix_initialize_n(&s_copy,s.height,s.width);
  copy_matrix(s_copy,s);
  matrix_add_scalar(s_copy,0.0000001);  
  matrix_sqrt(s_copy,0);
  matrix_div_scalar(s_copy,lr); 
  matrices_mul_n(s_copy,d);
  matrices_sub_n(w,s_copy);
}
  




double optimizer(Image_t* batch_images,int batch, int classes, double lr,double beta1, double beta2, Filter_t* filters,Matrix_t* weights,Matrix_t* biases, Filter_t* filters_grad,Matrix_t* weights_grad,Matrix_t* biases_grad, int itr,int* labels,int conv_s, int pool_s,int f,int pool_f,Filter_t* dfilters,Matrix_t* dbiases,Matrix_t* dweights,int* correct){
  
  double cost=0;
  // initialize gradients and momentum,RMS params
  
  
  Filter_t tot_f1;
  
  initializeFilter(&tot_f1,filters[0].depth_out,filters[0].depth_in,filters[0].f);
  
  Matrix_t tot_b1;
  matrix_initialize_n(&tot_b1,biases[0].height,biases[0].width);
  


  Filter_t tot_f2;
  initializeFilter(&tot_f2,filters[1].depth_out,filters[1].depth_in,filters[1].f);
  
  Matrix_t tot_b2;
  matrix_initialize_n(&tot_b2,biases[1].height,biases[1].width);
  

  Matrix_t tot_w3;
  matrix_initialize_n(&tot_w3,weights[0].height,weights[0].width);
  
  Matrix_t tot_w4;
  matrix_initialize_n(&tot_w4,weights[1].height,weights[1].width);

  /*
  for(int i = 0;i<tot_w3.height;i++){
    for(int j = 0;j<tot_w3.width;j++){
      tot_w3.matrix[i][j]=0;
    }
  } 
  
  
  for(int i = 0;i<tot_w4.height;i++){
    for(int j = 0;j<tot_w4.width;j++){
      tot_w4.matrix[i][j]=0;
    }
  } */
  Matrix_t tot_b3;
  matrix_initialize_n(&tot_b3,biases[2].height,biases[2].width);
  
  Matrix_t tot_b4;
  matrix_initialize_n(&tot_b4,biases[3].height,biases[3].width);
  /*printf("%s %f\n","f[0]",filters[1].filter[0][0].matrix[0][0]);
  printf("%s %f\n","f[1]",filters[1].filter[0][0].matrix[0][1]);
  printf("%s %f\n","f[2]",filters[1].filter[0][0].matrix[0][2]);
  printf("%s %f\n","f[3]",filters[1].filter[0][0].matrix[0][3]);

  printf("%s %f\n","w[0]",weights[0].matrix[0][0]);
  printf("%s %f\n","w[1]",weights[0].matrix[0][1]);
  printf("%s %f\n","w[2]",weights[0].matrix[0][2]);
  printf("%s %f\n","w[3]",weights[0].matrix[0][3]);*/
  /////
  for(int i =0;i<batch;i++){
    int** y = (int **)malloc( 2*sizeof(int*));
    y[0]=(int *)malloc(sizeof(int));
    y[1]=(int *)malloc(sizeof(int));
    
    if(labels[i]==0){

      y[0][0]=1;

      y[1][0]=0;

    }else{

      y[0][0]=0;

      y[1][0]=1;

    }


    int n_w=2;
    double*** grad_w = (double ***)malloc(n_w*sizeof(double**));
    int n_f=2;
    double**** grad_f = (double ****)malloc(n_f*sizeof(double***));  
      
    int n_b=4;
    double**** grad_b = (double ****)malloc(n_b*sizeof(double***));  
  Filter_t copy_df1;
  initializeFilter(&copy_df1,dfilters[0].depth_out,dfilters[0].depth_in,dfilters[0].f);
  copy_filter(copy_df1,dfilters[0]);
  Matrix_t copy_db1;
  matrix_initialize_n(&copy_db1,dbiases[0].height,dbiases[0].width);
  copy_matrix(copy_db1,dbiases[0]);

  Filter_t copy_df2;
  initializeFilter(&copy_df2,dfilters[1].depth_out,dfilters[1].depth_in,dfilters[1].f);
  copy_filter(copy_df2,dfilters[1]);
  Matrix_t copy_db2;
  matrix_initialize_n(&copy_db2,dbiases[1].height,dbiases[1].width);
  copy_matrix(copy_db2,dbiases[1]);

  Matrix_t copy_dw3;
  matrix_initialize_n(&copy_dw3,dweights[0].height,dweights[0].width);
  
 
  Matrix_t copy_dw4;
  matrix_initialize_n(&copy_dw4,dweights[1].height,dweights[1].width);

  /*
  for(int i = 0;i<copy_dw3.height;i++){
    for(int j = 0;j<copy_dw3.width;j++){
      copy_dw3.matrix[i][j]=1;
    }
  } 
  
  copy_matrix(copy_dw3,dweights[0]);
  for(int i = 0;i<copy_dw4.height;i++){
    for(int j = 0;j<copy_dw4.width;j++){
      copy_dw4.matrix[i][j]=1;
    }
  } */
  copy_matrix(copy_dw4,dweights[1]);
  Matrix_t copy_db3;
  matrix_initialize_n(&copy_db3,dbiases[2].height,dbiases[2].width);
  copy_matrix(copy_db3,dbiases[2]);
  Matrix_t copy_db4;
  matrix_initialize_n(&copy_db4,dbiases[3].height,dbiases[3].width);    
  copy_matrix(copy_db4,dbiases[3]);
  int predicted;
    double loss=apply_conv(batch_images[i], y, classes,f,conv_s,filters[0],biases[0],pool_f,pool_s,filters[1],biases[1],weights[0],weights[1],grad_w,grad_f,grad_b,biases[2],biases[3],copy_df1,copy_df2,copy_db1,copy_db2,copy_db3,copy_db4,copy_dw3,copy_dw4,&predicted);
   if(predicted==labels[i] )(*correct)++;
    cost+=loss;

    for(int i = 0;i<copy_df1.depth_out;i++){
    for(int j = 0;j<copy_df1.depth_in;j++){  
        
        matrices_add_n(tot_f1.filter[i][j],copy_df1.filter[i][j]);
        
      }
    }  
    for(int i = 0;i<copy_df2.depth_out;i++){
    for(int j = 0;j<copy_df2.depth_in;j++){  
        matrices_add_n(tot_f2.filter[i][j],copy_df2.filter[i][j]);
        
      }
    }
    
    matrices_add_n(tot_w3,copy_dw3);
    matrices_add_n(tot_w4,copy_dw4);
    matrices_add_n(tot_b1,copy_db1);
    matrices_add_n(tot_b2,copy_db2);
    matrices_add_n(tot_b3,copy_db3);
    matrices_add_n(tot_b4,copy_db4);
     matrix_initialize_n_free(copy_db1);
  matrix_initialize_n_free(copy_db2);
  matrix_initialize_n_free(copy_db3);
  matrix_initialize_n_free(copy_db4);
  matrix_initialize_n_free(copy_dw3);
  matrix_initialize_n_free(copy_dw4);
  initializeFilterFree(copy_df1);
  initializeFilterFree(copy_df2);
    
  }


  
  /*
  f1 -= (lr /np.sqrt(s1+1e-7))*df1 # combine momentum and RMSProp to perform update with Adam
  
  bs1 = beta2*bs1 + (1-beta2)*(db1)**2
  b1 -= (lr/np.sqrt(bs1+1e-7))*db1
 */
 
  adam_filter(filters[0],filters_grad[1], tot_f1,  beta1,  beta2,  lr,0);
  adam_matrix(biases[0],biases_grad[1], tot_b1,  beta1,  beta2,  lr,0);
    
  adam_filter(filters[1],filters_grad[3], tot_f2,  beta1,  beta2,  lr,0);
  adam_matrix(biases[1],biases_grad[3], tot_b2,  beta1,  beta2,  lr,0);
  
  adam_matrix(weights[0],weights_grad[1], tot_w3,  beta1,  beta2,  lr,0);
  adam_matrix(biases[2],biases_grad[5], tot_b3,  beta1,  beta2,  lr,0);
  
  
  adam_matrix(weights[1],weights_grad[3], tot_w4,  beta1,  beta2,  lr,0);
  adam_matrix(biases[3],biases_grad[7], tot_b4,  beta1,  beta2,  lr,0);
  
  printf("%f\n",cost/batch );
 
  matrix_initialize_n_free(tot_b1);
  matrix_initialize_n_free(tot_b2);
  matrix_initialize_n_free(tot_b3);
  matrix_initialize_n_free(tot_b4);
  matrix_initialize_n_free(tot_w3);
  matrix_initialize_n_free(tot_w4);

  initializeFilterFree(tot_f1);
  initializeFilterFree(tot_f2);

  return cost;


}

void  load_images_from_memory(Image_t* images_train,uint16_t tot_img_train,char* loc ,uint16_t width, uint16_t height, uint16_t image_channels){
  for(int i = 0;i<tot_img_train;i++){
    configuracion_t config;
    config.dst.width = 0;
    config.bits_src = 32;
    config.bits_dst = 32;
    config.es_video = false;
    config.verbose = false;
    config.frames = false;
    config.nombre = false;
    config.cant_iteraciones = 1;
    config.archivo_entrada = NULL;
    config.archivo_entrada_2 = NULL;
    config.carpeta_salida = ".";
    config.extra_archivo_salida = "";
    char buf[33];
    snprintf(buf, 33, loc, i); // puts string into buffer
    //printf("%s\n", buf); // outputs so you can see it
    config.archivo_entrada=buf;
    imagenes_abrir(&config);
    imagenes_flipVertical(&(&config)->src, src_img);
    buffer_info_t info = (&config)->src;
    uint8_t *src =  (uint8_t*)info.bytes;
    bgra_t* src_matrix = (bgra_t*)src;
    Image_t image__; 
    image_to_matrix(src_matrix,&image__,image_channels,height,width);
    images_train[i]= image__;
    free(src_matrix);

    }
}

void fill_filter(Filter_t filt, char* fn){
    FILE *fp = fopen(fn, "r");
    if (fp == NULL)
    {
        printf("Error: could not open file %s", fn);
        
    }

    // reading line by line, max 256 bytes
    const unsigned MAX_LENGTH = 256;
    char buffer[MAX_LENGTH];
    int a=0;
    for(int i=0;i<filt.depth_out;i++){
      for(int j=0;j<filt.depth_in;j++){
        for(int k=0;k<filt.f;k++){
          for(int l=0;l<filt.f;l++){
            fgets(buffer, MAX_LENGTH, fp);
            filt.filter[i][j].matrix[k][l]=strtod(buffer, NULL);
            //printf("%f %d\n",strtod(buffer, NULL),a++);
          }
        }
      }
    }
  
    // close the file
    fclose(fp);
}

void fill_matrix(Matrix_t matrix, char* fn){
    FILE *fp = fopen(fn, "r");
    if (fp == NULL)
    {
        printf("Error: could not open file %s", fn);
        
    }
    int a=0;
    // reading line by line, max 256 bytes
    const unsigned MAX_LENGTH = 256;
    char buffer[MAX_LENGTH];
    for(int i=0;i<matrix.height;i++){
      for(int j=0;j<matrix.width;j++){
        fgets(buffer, MAX_LENGTH, fp);
        matrix.matrix[i][j]=strtod(buffer, NULL);
        //printf("%f %d\n",matrix.matrix[i][j],a++);
      }
    }
  
    // close the file
    fclose(fp);
}
int main( int argc, char** argv ) {

  



  uint16_t tot_img_train_cat = 10;
  Image_t* images_train_cat = (Image_t*)malloc(tot_img_train_cat*sizeof(Image_t));
  
  uint16_t tot_img_train_dog = 10;
  Image_t* images_train_dog = (Image_t*)malloc(tot_img_train_dog*sizeof(Image_t));
  
  uint16_t number_filter2=8;
  uint16_t conv_s =1;
  uint16_t pool_s =2;
  uint16_t f = 3;
  uint16_t pool_f=2;
  uint16_t image_channels = 3;
  uint16_t image_dim = 150;
  uint16_t number_filter  = 8 ;
  uint16_t depth_filter   = 3 ;
  uint16_t classes = 2;
  double lr = 0.0001;
  double beta1=0.95;
  double beta2=0.9;
  
    
  load_images_from_memory( images_train_dog,tot_img_train_dog,"dogs/%d.bmp",image_dim,image_dim,image_channels);
  load_images_from_memory( images_train_cat,tot_img_train_cat,"cats/%d.bmp",image_dim,image_dim,image_channels);



  /*
  INICIO PARAMETROS
  */
  Filter_t f1;
  initializeFilter(&f1,number_filter,image_channels,f);
  fill_filter(f1, "f1.txt");
  Matrix_t b1;
  matrix_initialize_n(&b1,number_filter,1);
  
  Filter_t dfilt1;
  initializeFilter(&dfilt1,number_filter,image_channels,f);
  
  Matrix_t dbias1;
  matrix_initialize_n(&dbias1,number_filter,1);
  
  
  Filter_t f2;
  initializeFilter(&f2,number_filter2,number_filter,f);
  fill_filter(f2, "f2.txt");
  Matrix_t b2;
  matrix_initialize_n(&b2,number_filter2,1);


  Filter_t dfilt2;
  initializeFilter(&dfilt2,number_filter2,number_filter,f);

  Matrix_t dbias2;
  matrix_initialize_n(&dbias2,number_filter2,1);
  
  Matrix_t w3;
  matrix_initialize_n(&w3,128,42632);
  fill_matrix(w3,"w3.txt");
  
  Matrix_t w4;
  matrix_initialize_n(&w4,2,128);
  fill_matrix(w4,"w4.txt");
  Matrix_t dw3;
  matrix_initialize_n(&dw3,128,42632);
  
  Matrix_t dw4;
  matrix_initialize_n(&dw4,2,128);
  
  for(int i = 0;i<w3.height;i++){
    for(int j = 0;j<w3.width;j++){
      dw3.matrix[i][j]=1;
    }
  } 
  
  
  for(int i = 0;i<w4.height;i++){
    for(int j = 0;j<w4.width;j++){
      dw4.matrix[i][j]=1;
    }
  } 


  Matrix_t b3;
  matrix_initialize_n(&b3,128,1);
  

  Matrix_t db3;
  matrix_initialize_n(&db3,128,1);
  

  Matrix_t b4;
  matrix_initialize_n(&b4,2,1);


  Matrix_t db4;
  matrix_initialize_n(&db4,2,1);
  
   /*Armo vectores de parametros*/  
  
  Matrix_t* biases=(Matrix_t*)malloc( 4*sizeof(Matrix_t));
  Matrix_t* weights=(Matrix_t*)malloc( 2*sizeof(Matrix_t));
  Filter_t* filters=(Filter_t*)malloc( 2*sizeof(Filter_t));
  Matrix_t* dbiases=(Matrix_t*)malloc( 4*sizeof(Matrix_t));
  Matrix_t* dweights=(Matrix_t*)malloc( 2*sizeof(Matrix_t));
  
  Filter_t* dfilters=(Filter_t*)malloc( 2*sizeof(Filter_t));
  
  biases[0]=b1;
  biases[1]=b2;
  biases[2]=b3;
  biases[3]=b4;
  weights[0]=w3;
  weights[1]=w4;
  filters[0]=f1;
  filters[1]=f2;

  dbiases[0]=dbias1;
  dbiases[1]=dbias2;
  dbiases[2]=db3;
  dbiases[3]=db4;
  dfilters[0]=dfilt1;
  dfilters[1]=dfilt2;
  dweights[0]=dw3;
  dweights[1]=dw4;
  
  
  Matrix_t* biases_grad=(Matrix_t*)malloc( 2*4*sizeof(Matrix_t));
  Matrix_t* weights_grad=(Matrix_t*)malloc( 2*2*sizeof(Matrix_t));
  Filter_t* filters_grad=(Filter_t*)malloc( 2*2*sizeof(Filter_t));
  
  Filter_t v1;
  initializeFilter(&v1,number_filter,image_channels,f);

  Filter_t s1;
  initializeFilter(&s1,number_filter,image_channels,f);

  filters_grad[0]=v1;
  filters_grad[1]=s1;
  
 
  
  Filter_t v2;
  initializeFilter(&v2,number_filter2,number_filter,f);

  Filter_t s2;
  initializeFilter(&s2,number_filter2,number_filter,f);

  filters_grad[2]=v2;
  filters_grad[3]=s2;

  Matrix_t v3;
  matrix_initialize_n(&v3,128,42632);

  Matrix_t s3;
  matrix_initialize_n(&s3,128,42632);
  
  weights_grad[0]=v3;
  weights_grad[1]=s3;


  Matrix_t v4;
  matrix_initialize_n(&v4,2,128);

  Matrix_t s4;
  matrix_initialize_n(&s4,2,128);

  weights_grad[2]=v4;
  weights_grad[3]=s4;
  
  Matrix_t bv1;
  matrix_initialize_n(&bv1,number_filter,1);

  Matrix_t bs1;
  matrix_initialize_n(&bs1,number_filter,1);
  
  biases_grad[0]=bv1;
  biases_grad[1]=bs1;
  
  Matrix_t bv2;
  matrix_initialize_n(&bv2,number_filter2,1);

  Matrix_t bs2;
  matrix_initialize_n(&bs2,number_filter2,1);
  biases_grad[3]=bv2;
  biases_grad[3]=bs2;
  
  
  Matrix_t bv3;
  matrix_initialize_n(&bv3,128,1);
  
  Matrix_t bs3;
  matrix_initialize_n(&bs3,128,1);
  
  biases_grad[4]=bv3;
  biases_grad[5]=bs3;

  Matrix_t bv4;
  matrix_initialize_n(&bv4,2,1);
  
  Matrix_t bs4;
  matrix_initialize_n(&bs4,2,1);
  
  biases_grad[6]=bv4;
  biases_grad[7]=bs4;


  
    
  /*
  FIN PARAMETROS 
  */
  
  int itr;
  unsigned int epochs=20;
  unsigned int batch = 10;
  unsigned int batch_iterations= (tot_img_train_cat+tot_img_train_dog)/batch;
  double cost[epochs*batch];
  for(int i=0;i<epochs;i++){
    printf("%s %d\n","Epcoch: ",i);
    int vektor[tot_img_train_cat+tot_img_train_dog];
    int labels[batch];
    int contador=0;    
    random_batch(tot_img_train_cat+tot_img_train_dog, vektor,tot_img_train_cat+tot_img_train_dog);
    int correct_total=0;
    for(int j=0;j<batch_iterations;j++){
      
      Image_t* batch_images = (Image_t*)malloc(batch*sizeof(Image_t));
      printf("%s %d\n","Batch: ",j);
      for(int k=0;k<batch;k++){
        //Cats from 0 to 999. Dogs from 1000 to 1999
        if(vektor[contador]>=tot_img_train_dog){
          batch_images[k]=images_train_dog[vektor[contador]-tot_img_train_dog];
          labels[k]=1;
        }
        if(vektor[contador]<tot_img_train_cat){
          batch_images[k]=images_train_cat[vektor[contador]];
          labels[k]=0;
        }
        contador++;
        itr=j*k;//Aca tengo que guardar el costo
        
      }
      int correct=0;
      double loss_tot =optimizer(batch_images,batch, classes, lr,beta1, beta2, filters,weights,biases, filters_grad,weights_grad,biases_grad,itr,labels, conv_s, pool_s,f,pool_f,dfilters,dbiases,dweights,&correct);
      correct_total+=correct;
      cost[i*j]=0;
      free(batch_images);  
      
    
    }
  
  printf("%s %f\n","Acc",(double)correct_total/(tot_img_train_cat+tot_img_train_dog));
  }


  

  return 0;
}
// gcc -o ejec multiplicarMatrices.c -lm 
// ./ejec aa -i asm name.bmp


//https://stackoverflow.com/questions/12747731/ways-to-create-dynamic-matrix-in-c







