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



void dot(
    double **matrix_A,
    double **matrix_B,
    int width_A,
    int height_A,
    int width_B,
    int height_B,
    double **res
    )
{
    assert(width_A==height_B && "The dimension of the matrices do not match");

    for(int i = 0;i< height_A;i++){
        for(int j = 0;j< width_B;j++){
            double sum = 0;
            for(int k = 0;k< width_A;k++){
                 //printf("%f\n", matrix_A[i*width_A+ k]); 
                 //printf("%f\n", matrix_B[k*width_B +j]);
                sum+= matrix_A[i][k]*matrix_B[k][j];
            }   
            //printf("%Lf\n",sum);
            res[i][j]=sum;  
            
                
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

void matrices_add(
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
          matrix_A[i][j] = matrix_A[i][j]+matrix_B[i][j];
        }
    }

   
}

void transpose(
  double **matrix_A,
  int width_A,
  int height_A,
  double **transposed){
  for(int i = 0;i< height_A;i++){
        for(int j = 0;j< width_A;j++){
          transposed[j][i] = matrix_A[i][j];
        }
    }




}

void matrices_mul(
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
          matrix_A[i][j] = matrix_A[i][j]*matrix_B[i][j];
        }
    }


}


double matrices_add_all_elements(
    double **matrix_A,
    int width_A,
    int height_A){
    double res = 0;
    for(int i = 0;i< height_A;i++){
        for(int j = 0;j< width_A;j++){
         res+=matrix_A[i][j];
        }
    } 
    return res;
}

void matrices_relu( 
    double ***matrix,
    unsigned int width,
    unsigned int height,
    unsigned int image_channels){
    for(int k = 0;k< image_channels;k++){
      for(int i = 0;i< height;i++){
          for(int j = 0;j< width;j++){
            if (matrix[k][i][j]<=0)matrix[k][i][j] =0;
          }
      }
    }


}

void matrices_flatten(
  double*** matrix, 
  int depth, 
  int width,
  int height,
  double **res){
  int counter = 0;
  for(int k = 0;k< depth;k++){
    for(int i = 0;i< height;i++){
          for(int j = 0;j< width;j++){
          res[counter][0]  = matrix[k][i][j];
          
          counter++;
          }
      }
  }
 

}


void matrices_reshape_from_flatten( 
  double **matrix, 
  int width_matrix,
  int height_matrix,
  int depth, 
  int width,
  int height,
  double ***res){
    int f = 0;
    int g = 0;
    for(int k = 0;k< depth;k++){
      for(int i = 0;i< height;i++){
        for(int j = 0;j< width;j++){
          res[k][i][j]=matrix[f][g];
          g++;
          if(g>=width){
             g=0;
             f++; 
          }
        }
      }
    }
  }


void initializeFilter(
  unsigned int f,
  unsigned int number_filter,
  unsigned int number_filter2,
  
  double**** filt){
    for(int i = 0;i<number_filter;i++){
      filt[i] = (double ***)malloc( number_filter2*sizeof(double**));
      for(int j = 0;j<number_filter2;j++){  
          filt[i][j] = (double **)malloc( f*sizeof(double*));
          for(int k = 0;k< f;k++){
            filt[i][j][k] = (double *)malloc( f*sizeof(double));
            for(int u = 0;u< f;u++){
               filt[i][j][k][u]=1;
            }
      

      }

         
      } 
      
    }

}
void matrix_initialize(
  unsigned int height,
  unsigned int width,
  double** matrix){

  for(int i = 0;i< height;i++){
      
    matrix[i]= (double *)malloc( width*sizeof(double));
    for(int j = 0;j< width;j++){
      matrix[i][j]=0;
      
    }
  }
      
}
void matrix_initialize_free(
  unsigned int height,
  double** matrix){
  for(int i = 0;i< height;i++)free(matrix[i]);
  }

void image_initialize(
  unsigned int depth,
  unsigned int height,
  unsigned int width,
  double*** image){
  for(int i = 0;i< depth;i++){
      double** temp = (double **)malloc( height*sizeof(double*));

      matrix_initialize(height,width,temp);
      image[i]=temp;
      
    }

  }
  
void image_initialize_free(
  unsigned int depth,
  unsigned int height,
  double*** image){
  for(int i = 0;i< depth;i++){
    for(int j = 0;j< height;j++){
      free(image[i][j]);
    }
    free(image[i]);
    }

  }
  

void image_to_matrix(//RGB
  bgra_t *src_matrix,
  int width,
  int height,
  double*** res){
  int n_channels=3;
  double* pixels = (double *)malloc(n_channels*width*height*sizeof(double));
  int c =0;
  for(int i  = 0; i<height;i++){
      for(int j  = 0; j<width;j++){
        pixels[c]=src_matrix[i*width+ j].r;
        pixels[c+1]=src_matrix[i*width+ j].g;
        pixels[c+2]=src_matrix[i*width+ j].b;
        c+=3;
      }
  }

 
  c=0;
  for(int i = 0;i<n_channels;i++){
    res[i]=(double **)malloc((height+10)*sizeof(double*));
      
    for (int j = 0; j < height; j++){
   
      res[i][j]=(double *)malloc((width+1)*sizeof(double));
      for (int k = 0; k < width; k++){
        res[i][j][k]=pixels[c];
        c++;
      }
    }
  }
  free(pixels);

}
void slice(
  double ***matrix_A,
  int width_A,
  int height_A,
  int start_y,
  int start_x,
  int step,
  int depth,
  double ***res){

   assert(start_x+step<=width_A && "Limit out of bounds");
   assert(start_y+step<=height_A && "Limit out of bounds");

   for(int t =0;t<depth;t++){ 
     res[t] = (double **)malloc(step *sizeof(double*));   
     int k = 0;
     for(int i = start_y;i<start_y+step;i++){
        int u=0;
        res[t][k] = (double *)malloc(step *sizeof(double)); 
        for(int j = start_x;j<start_x+step;j++){
            
            res[t][k][u] = matrix_A[t][i][j]; 
            
            u++;
            
        }    
        k++;
             
     }
   }
   

}

void slice_free(
  int start_y,
  int step,
  int depth,
  double ***res){
  for(int t =0;t<depth;t++){ 
    int k=0;
    for(int p = start_y;p<start_y+step;p++){
      free(res[t][k]); 
      k++;
    }
    free(res[t]);   
  }

}

void convolution(double ***image,double ****filt,double **bias,int s, int image_channels,int image_dim, int number_filter, int depth_filter,int f, int out_image,double ***res){
   assert(image_channels==depth_filter && "Dimensions of filter must match dimensions of input image");
   //FILE * fp;

   //fp = fopen ("ultima.txt", "w+");
 
   
   for(int i = 0;i< number_filter;i++){
    int curr_y = 0;
    int out_y = 0;
    
    while(curr_y + f <= image_dim){
      int curr_x=0;
      int out_x = 0;
      while(curr_x + f <= image_dim){
          double sum = 0;
         
          double*** slice_image = (double ***)malloc(image_channels*sizeof(double**));   
          slice(image,image_dim,image_dim,curr_y,curr_x,f,image_channels,slice_image);
          for(int j =0;j<image_channels;j++){
            matrices_mul(slice_image[j],filt[i][j],f,f,f,f);
            sum+= matrices_add_all_elements(slice_image[j],f,f);
          }
          slice_free(curr_y,f,image_channels,slice_image);
          
          free(slice_image);
        
        sum+=bias[i][0];
          
        
        //fprintf(fp, "%d\n",(int)sum);
        res[i][out_y][out_x] =sum;
        curr_x+=s;
        out_x+=1;
      }
      curr_y+=s;
      out_y+=1;
    }
   }

   //fclose(fp);

}




void maxpool(double ***image,unsigned int f, unsigned int s, unsigned int image_channels,unsigned int width, unsigned int height,unsigned int image_dim,double ***res){
    ///FILE * fp;

    //fp = fopen ("d.txt", "w+");
    for(unsigned int i = 0;i<image_channels;i++){
        int curr_y=0;
        int out_y=0;
        while(curr_y+f<=image_dim){
            int curr_x =0;
            int out_x  =0;
            while(curr_x+f<=image_dim){
                double*** slice_image = (double ***)malloc(image_channels *sizeof(double**));   
                slice(image,image_dim,image_dim,curr_y,curr_x,f,image_channels,slice_image);
                double max=slice_image[i][0][0];
                for(int j=0;j<f;j++){
                    for(int k=0;k<f;k++){
                        if(max<slice_image[i][j][k])max=slice_image[i][j][k];
                    }  
                }
                res[i][out_y][out_x]=max;
                //fprintf(fp, "%d\n",(int)max);
                slice_free(curr_y,f,image_channels,slice_image);
          
                free(slice_image);
                curr_x+=s;
                out_x+=1;

            }
            curr_y+=s;
            out_y+=1;

      
        }

    }
    //
  
  
  //fclose(fp);  
}

void backwardConvolution(double ***conv_in,double ***dconv_prev,double ****dfilt,double ****filt,double **dbias,int s, int image_channels,int image_dim, int number_filter, int depth_filter,int f,int conv_in_dim,double ***dout){
  assert(image_channels==depth_filter && "Dimensions of filter must match dimensions of input image");
  
  ////FILE * fp;
    ////fp = fopen ("backConv_.txt", "w+");
   for(int i = 0;i< number_filter;i++){
    int curr_y = 0;
    int out_y = 0;
    
    while(curr_y + f <= image_dim){
      int curr_x=0;
      int out_x = 0;
      while(curr_x + f <= image_dim){
          
        
        for(int j = 0;j<depth_filter;j++){
            for(int k = 0;k<f;k++){
              for(int m = 0;m<f;m++){

                dfilt[i][j][k][m]= dconv_prev[i][out_y][out_x]*conv_in[j][curr_y+k][curr_x+m];
              }
            }
          }
        for(int j = 0;j<image_channels;j++){
          for(int k = 0;k<f;k++){
            for(int m = 0;m<f;m++){
              
              //printf("%d %d %d",j,k,m,image_channels);
              dout[j][curr_y+k][curr_x+m]+=dconv_prev[i][out_y][out_x]*filt[i][j][k][m];
            }
          }
        }
        
       
        curr_x+=s;
        out_x+=1;
      }
      curr_y+=s;
      out_y+=1;
    }
    dbias[i][0]+=matrices_add_all_elements(dconv_prev[i],conv_in_dim,conv_in_dim);

    
   }

   //fclose(fp);
  

}



void backwardMaxpool(double ***dpool,double ***image,unsigned int f, unsigned int s, unsigned int image_channels,unsigned int width, unsigned int height,unsigned int image_dim,double ***res){
    //FILE * fp;
    //fp = fopen ("backwardMaxpool.txt", "w+");
    for(unsigned int i = 0;i<image_channels;i++){
        int curr_y=0;
        int out_y=0;
        while(curr_y+f<=image_dim){
            int curr_x =0;
            int out_x  =0;
            while(curr_x+f<=image_dim){
                double*** slice_image = (double ***)malloc(image_channels *sizeof(double**));   
                slice(image,image_dim,image_dim,curr_y,curr_x,f,image_channels,slice_image);
                double max=slice_image[i][0][0];
                int a=0;
                int b=0;
                for(int j=0;j<f;j++){
                    for(int k=0;k<f;k++){
                        if(max<slice_image[i][j][k]){
                          max=slice_image[i][j][k];
                          a=j;
                          b=k;
                        }
                    }  
                }
                res[i][out_y+a][out_x+b]=dpool[i][out_y][out_x];
                              
               // fprintf(fp, "%d\n",(int)dpool[i][out_y][out_x]);
                slice_free(curr_y,f,image_channels,slice_image);
          
                free(slice_image);
                curr_x+=s;
                out_x+=1;

            }
            curr_y+=s;
            out_y+=1;

      
        }


    }

  //fclose(fp);  
}

void softmax(double** x,int size){
  double max= x[0][0];
  double sum=0;
  for(int i=0;i<size;i++){
    if(max<x[i][0])max=x[i][0];
   
  }

  for(int i=0;i<size;i++){
    x[i][0]=exp(x[i][0]-max);
    sum+=x[i][0];
  }
  for(int i=0;i<size;i++){
    x[i][0]=x[i][0]/sum;
    
  }
}

double categoricalCrossEntropy(double** probs,int** label,int size){ 
  double res=0;
  for(int i=0;i<size;i++)res+=(label[i][0]*log(probs[i][0]));
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




double apply_conv(int image_channels, int image_dim, double*** image, int** labels, int number_filter, int classes,int f,int conv_s,double**** filt1 ,double** bias1,int depth_filter,int number_filter2, int pool_f,int pool_s,double**** filt2,double**bias2,double**w3,double**w4,double*** grad_w,double**** grad_f,double**** grad_b,double** b3,double** b4,double**** dfilt1,double**** dfilt2,double** dbias1,double** dbias2){
    
    ///double*** image = (double ***)malloc( image_channels*sizeof(double**));

    ///image_to_matrix(src_matrix,image_dim,image_dim,image);
    /*START CONV1*/
    double*** conv1 = (double ***)malloc( number_filter*sizeof(double**));
    int conv1_image = (int)((image_dim - f)/conv_s)+1;
    image_initialize(number_filter,conv1_image,conv1_image,conv1);
    convolution(image,filt1,bias1, conv_s,  image_channels, image_dim,  number_filter,  depth_filter, f,  conv1_image,conv1);
    matrices_relu( conv1,conv1_image,conv1_image,number_filter);
    /*END CONV1*/
    
    /*START CONV2*/
    double*** conv2 = (double ***)malloc( number_filter2*sizeof(double**));
    
    int conv2_image = (int)((conv1_image - f)/conv_s)+1;
    image_initialize(number_filter2,conv2_image,conv2_image,conv2);
    convolution(conv1,filt2,bias2, conv_s,  number_filter, conv1_image,  number_filter2,  number_filter2, f,  conv2_image,conv2);
    matrices_relu( conv2,conv2_image,conv2_image,number_filter2);
    
    /*END CONV2   */ 


    unsigned int height=(unsigned int)((conv2_image - pool_f)/pool_s)+1;
    unsigned int width=(unsigned int)((conv2_image - pool_f)/pool_s)+1;
    
    
    double*** pooled = (double ***)malloc( number_filter2*sizeof(double**));
    image_initialize(number_filter2,height,width,pooled);
    maxpool(conv2, 2,  pool_s, number_filter2,width, height,conv2_image,pooled);
    
    double** fc = (double **)malloc( number_filter2*width*height*sizeof(double*));

    matrix_initialize(number_filter2*width*height,1,fc);
    matrices_flatten( pooled, number_filter2, width,height,fc);
    double** z = (double **)malloc(128*sizeof(double*));
    matrix_initialize(128,1,z);
    dot(w3,fc,42632,128,1,42632,z);
    matrices_add(z,b3,1,128,1,128);  

    //RELU
    for(int i=0;i<128;i++){
      for(int j=0;j<1;j++){
       if(z[i][j]<=0)z[i][j]=0;
          
      }    
    }
    
    
    double** last = (double **)malloc( 2*sizeof(double*));
    
    matrix_initialize(2,1,last);
    
    dot(w4,z,128,2,1,128,last);
    matrices_add(last,b4,1,2,1,2);  ;
    softmax(last,2);
   
    double loss = categoricalCrossEntropy(last,labels,2);
  /*  ################################################
      ############# Backward Operation ###############
      ################################################ */ 
    double** dout = (double **)malloc( classes*sizeof(double*));
    for(int i=0;i<classes;i++){
      dout[i]=(double *)malloc(sizeof(double));
      dout[i][0]=last[i][0]-(double)labels[i][0];
      
    }
    double** z_transpose = (double **)malloc(1*sizeof(double*));
    matrix_initialize(1,128,z_transpose);
    transpose(z,1,128,z_transpose);
    double** dw4 = (double **)malloc( 2*sizeof(double*));

    matrix_initialize(2,128,dw4);
    
    dot(dout,z_transpose,1,2,128,1,dw4);
    //TODO faltaria hacer db4 pero parecer ser lo mismo que dout
    
    double** w4_transpose = (double **)malloc(128*sizeof(double*));
    matrix_initialize(128,2,w4_transpose);
    transpose(w4,128,2,w4_transpose);
    
    double** dz = (double **)malloc( 128*sizeof(double*));
    matrix_initialize(128,1,dz);

    dot(w4_transpose,dout,2,128,1,2,dz);
    for(int i=0;i<128;i++){
      for(int j=0;j<1;j++){
       if(dz[i][j]<=0)dz[i][j]=0;
          
      }    
    }
    double** fc_transpose = (double **)malloc(1*sizeof(double*));
    matrix_initialize(1,number_filter2*width*height,fc_transpose);
    transpose(fc,1,number_filter2*width*height,fc_transpose);

    double** dw3 = (double **)malloc( 128*sizeof(double*));
    matrix_initialize(128,number_filter2*width*height,dw3);

    dot(dz,fc_transpose,1,128,number_filter2*width*height,1,dw3);
    
    //TODO faltaria hacer db3 pero parecer ser lo mismo que dz
    
    double** w3_transpose = (double **)malloc(number_filter2*width*height*sizeof(double*));
    matrix_initialize(number_filter2*width*height,128,w3_transpose);
    transpose(w3,number_filter2*width*height,128,w3_transpose);
    
    double** dfc = (double **)malloc( number_filter2*width*height*sizeof(double*));
    matrix_initialize(number_filter2*width*height,1,dfc);
    
    dot(w3_transpose,dz,128,number_filter2*width*height,1,128,dfc);
    
    double*** dpool = (double ***)malloc( number_filter2*sizeof(double**));
    image_initialize(number_filter2,height,width,dpool);
    

    matrices_reshape_from_flatten(dfc, 1,number_filter2*width*height,number_filter2, width,height,dpool);
    //Hasta aca anda re piola
    double*** dconv2 = (double ***)malloc( number_filter2*sizeof(double**));
    
    image_initialize(number_filter2,conv2_image,conv2_image,dconv2);
    backwardMaxpool(dpool,conv2,pool_f,pool_s,number_filter2,conv2_image,conv2_image,conv2_image,dconv2);
    matrices_relu( dconv2,conv2_image,conv2_image,number_filter2);

    double*** dconv1 = (double ***)malloc( number_filter*sizeof(double**));
    
    image_initialize(number_filter,conv1_image,conv1_image,dconv1);
    backwardConvolution(conv1,dconv2,dfilt2,filt2,dbias2,conv_s, number_filter,conv1_image, number_filter2, number_filter2,f,conv2_image, dconv1);
    matrices_relu( dconv1,conv1_image,conv1_image,number_filter);

    double*** dimage = (double ***)malloc( image_channels*sizeof(double**));
    
    image_initialize(image_channels,image_dim,image_dim,dimage);
    
    backwardConvolution(image,dconv1,dfilt1,filt1,dbias1,conv_s, image_channels,image_dim, number_filter, image_channels,f,conv1_image, dimage);
    
    
    
    image_initialize_free(number_filter,conv1_image,conv1);
    image_initialize_free(number_filter2,conv2_image,conv2);
    image_initialize_free(number_filter2,height,pooled);
    image_initialize_free(number_filter,conv1_image,dconv1);
    image_initialize_free(number_filter2,conv2_image,dconv2);
    image_initialize_free(number_filter2,height,dpool );
    image_initialize_free(image_channels,image_dim,dimage);
    free(conv2);
    free(conv1);
    free(pooled);
    free(dconv1);
    free(dconv2);
    free(dpool);
    free(dimage);
    matrix_initialize_free(number_filter2*width*height,fc);
    free(fc);
    matrix_initialize_free(128,z);
    
    free(z);
    matrix_initialize_free(2,last);
    free(last);
    for(int i=0;i<classes;i++){free(dout[i]);}
    free(dout);
          
    matrix_initialize_free(number_filter2*width*height,dfc);
    free(dfc);
    matrix_initialize_free(number_filter2*width*height,w3_transpose);
    free(w3_transpose);
    matrix_initialize_free(128,dw3);
    free(dw3);
    matrix_initialize_free(1,fc_transpose);
    free(fc_transpose);
    matrix_initialize_free(128,dz);
    free(dz);
    matrix_initialize_free(128,w4_transpose);
    free(w4_transpose);
    matrix_initialize_free(2,dw4);
    free(dw4);
    matrix_initialize_free(1,z_transpose);
    free(z_transpose);
    return loss;
}

double optimizer(double**** batch_images,int batch, int classes, double lr,int image_dim, int image_channels, double beta1, double beta2, double***** filters,double*** weights,double*** biases, double***** filters_grad,double*** weights_grad,double*** biases_grad, int itr,int* labels,int number_filter2, int conv_s, int pool_s,int f,int pool_f,int number_filter,int depth_filter,double***** dfilters,double*** dbiases){
  double cost=0;
  // initialize gradients and momentum,RMS params
  //TODO Free memory here
  double**** tot_f1 = (double ****)malloc( number_filter*sizeof(double***));
  double** tot_b1 = (double **)malloc( number_filter*sizeof(double*));
  initializeFilter(f,number_filter,number_filter2,tot_f1);
  matrix_initialize(number_filter,1,tot_b1);
  
  double**** tot_f2 = (double ****)malloc( number_filter*sizeof(double***));
  double** tot_b2 = (double **)malloc( number_filter*sizeof(double*));
  initializeFilter(f,number_filter,number_filter2,tot_f2);
  matrix_initialize(number_filter2,1,tot_b2);


  double** tot_w3 = (double **)malloc( 128*sizeof(double*));
  matrix_initialize(128,42632,tot_w3);
  double** tot_w4 = (double **)malloc( 2*sizeof(double*));
  matrix_initialize(2,128,tot_w4);
  for(int i = 0;i<128;i++){
    for(int j = 0;j<42632;j++){
      tot_w3[i][j]=1;
    }
  } 
  for(int i = 0;i<2;i++){
    for(int j = 0;j<128;j++){
      tot_w4[i][j]=1;
    }
  } 

  double** tot_b3 = (double **)malloc( 128*sizeof(double*));
  matrix_initialize(128,1,tot_b3);
  double** tot_b4 = (double **)malloc( 2*sizeof(double*));
  matrix_initialize(2,1,tot_b4);
  /////
  for(int i =0;i<batch;i++){
    int** y = (int **)malloc( 2*sizeof(int*));
    y[0]=(int *)malloc(sizeof(int));
    y[1]=(int *)malloc(sizeof(int));
    
    if(labels[i]==0){

      y[0][0]=0;

      y[1][0]=1;

    }else{

      y[0][0]=1;

      y[1][0]=0;

    }



    int n_w=2;
    double*** grad_w = (double ***)malloc(n_w*sizeof(double**));
    int n_f=2;
    double**** grad_f = (double ****)malloc(n_f*sizeof(double***));  
      
    int n_b=4;
    double**** grad_b = (double ****)malloc(n_b*sizeof(double***));  
    double loss=apply_conv(image_channels, image_dim, batch_images[i], y,number_filter, classes,f,conv_s,filters[0],biases[0],depth_filter,number_filter2, pool_f,pool_s,filters[1],biases[1],weights[0],weights[1],grad_w,grad_f,grad_b,biases[2],biases[3],dfilters[0],dfilters[1],dbiases[0],dbiases[1]);
    cost+=loss;

    for(int i = 0;i<number_filter;i++){
    for(int j = 0;j<number_filter2;j++){  
        matrices_add(tot_f1[i][j],filters[0][i][j],f,f,f,f);
        
      }
    }  
    for(int i = 0;i<number_filter;i++){
    for(int j = 0;j<number_filter2;j++){  
        matrices_add(tot_f1[i][j],filters[1][i][j],f,f,f,f);
        
      }
    }
   
    matrices_add(tot_w3,weights[0],42632,128,42632,128);
    matrices_add(tot_w4,weights[1],2,128,2,128);
    matrices_add(tot_b1,biases[0],1,number_filter,1,number_filter);
    matrices_add(tot_b2,biases[1],1,number_filter2,1,number_filter2);
    matrices_add(tot_b3,biases[2],1,128,1,128);
    matrices_add(tot_b4,biases[3],1,2,1,2);
    
  
    free(grad_w);
    free(grad_f);
    free(grad_b);
    free(y);
  }
  return cost;


}



int main( int argc, char** argv ) {
  /*unsigned int tot_img_train_cat = 1000;
  double**** images_train_cat = (double****)malloc(tot_img_train_cat*sizeof(double***));
  
  unsigned int tot_img_train_dog = 1000;
  double**** images_train_dog = (double****)malloc(tot_img_train_dog*sizeof(double***));
  
  int number_filter2=8;
  int conv_s =1;
  int pool_s =2;
  int f = 3;
  int pool_f=2;
  int image_channels = 3;
  int image_dim = 150;
  int number_filter  = 8 ;
  int depth_filter   = 3 ;
  int classes = 2;
  double lr = 0.01;
  double beta1=0;
  double beta2=0;
  
  for(int i = 0;i<tot_img_train_cat;i++){
    
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
    snprintf(buf, 33, "cats_resized/cat.%d.bmp", i); // puts string into buffer
    //printf("%s\n", buf); // outputs so you can see it
    config.archivo_entrada=buf;
    imagenes_abrir(&config);
    imagenes_flipVertical(&(&config)->src, src_img);
    buffer_info_t info = (&config)->src;
    uint8_t *src =  (uint8_t*)info.bytes;
    bgra_t* src_matrix = (bgra_t*)src;
    double*** image__ = (double ***)malloc( image_channels*sizeof(double**));
    image_to_matrix(src_matrix,image_dim,image_dim,image__);
    images_train_cat[i]= image__; 
    free(src_matrix);
    
  }

  for(int i = 0;i<tot_img_train_dog;i++){
    
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
    snprintf(buf, 33, "dogs_resized/dog.%d.bmp", i); // puts string into buffer
    //printf("%s\n", buf); // outputs so you can see it
    config.archivo_entrada=buf;
    imagenes_abrir(&config);
    imagenes_flipVertical(&(&config)->src, src_img);
    buffer_info_t info = (&config)->src;
    uint8_t *src =  (uint8_t*)info.bytes;
    bgra_t* src_matrix = (bgra_t*)src;
    double*** image__ = (double ***)malloc( image_channels*sizeof(double**));
    image_to_matrix(src_matrix,image_dim,image_dim,image__);
    images_train_dog[i]= image__; 
    free(src_matrix);
    }
    

  /*
  INICIO PARAMETROS
  */
  /*
  double**** f1 = (double ****)malloc( number_filter*sizeof(double***));
  double** b1 = (double **)malloc( number_filter*sizeof(double*));
  initializeFilter(f,number_filter,number_filter2,f1);
  matrix_initialize(number_filter,1,b1);

  double**** dfilt1 = (double ****)malloc( number_filter*sizeof(double***));
  double** dbias1 = (double **)malloc( number_filter*sizeof(double*));
  initializeFilter(f,number_filter,number_filter2,dfilt1);
  matrix_initialize(number_filter,1,dbias1);
  
  double**** f2 = (double ****)malloc( number_filter*sizeof(double***));
  double** b2 = (double **)malloc( number_filter*sizeof(double*));
  initializeFilter(f,number_filter,number_filter2,f2);
  matrix_initialize(number_filter2,1,b2);

  double**** dfilt2 = (double ****)malloc( number_filter*sizeof(double***));
  double** dbias2 = (double **)malloc( number_filter*sizeof(double*));
  initializeFilter(f,number_filter,number_filter2,dfilt2);
  matrix_initialize(number_filter2,1,dbias2);
  


  double** w3 = (double **)malloc( 128*sizeof(double*));
  matrix_initialize(128,42632,w3);
  double** w4 = (double **)malloc( 2*sizeof(double*));
  matrix_initialize(2,128,w4);
  for(int i = 0;i<128;i++){
    for(int j = 0;j<42632;j++){
      w3[i][j]=1;
    }
  } 
  for(int i = 0;i<2;i++){
    for(int j = 0;j<128;j++){
      w4[i][j]=1;
    }
  } 

  double** b3 = (double **)malloc( 128*sizeof(double*));
  matrix_initialize(128,1,b3);
  double** b4 = (double **)malloc( 2*sizeof(double*));
  matrix_initialize(2,1,b4);
   /*Armo vectores de parametros*/ /*
  double*** biases=(double ***)malloc( 4*sizeof(double**));
  double*** weights=(double ***)malloc( 2*sizeof(double**));
  double***** filters=(double *****)malloc( 2*sizeof(double****));
  double*** dbiases=(double ***)malloc( 2*sizeof(double**));
  double***** dfilters=(double *****)malloc( 2*sizeof(double****));
  
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
  dfilters[0]=dfilt1;
  dfilters[1]=dfilt2;
 
  double*** biases_grad=(double ***)malloc( 2*4*sizeof(double**));
  double*** weights_grad=(double ***)malloc( 2*2*sizeof(double**));
  double***** filters_grad=(double *****)malloc( 2*2*sizeof(double****));
  
  double**** v1 = (double ****)malloc( number_filter*sizeof(double***));
  double**** s1 = (double ****)malloc( number_filter*sizeof(double***));
  filters_grad[0]=v1;
  filters_grad[1]=s1;
  double**** v2 = (double ****)malloc( number_filter*sizeof(double***));
  double**** s2 = (double ****)malloc( number_filter*sizeof(double***));
  filters_grad[2]=v2;
  filters_grad[3]=s2;
  double** v3 = (double **)malloc( 128*sizeof(double*));
  double** s3 = (double **)malloc( 128*sizeof(double*));
  weights_grad[0]=v3;
  weights_grad[1]=s3;
  double** v4 = (double **)malloc( 2*sizeof(double*));
  double** s4 = (double **)malloc( 2*sizeof(double*));
  weights_grad[2]=v4;
  weights_grad[3]=s4;
  double** bv1 = (double **)malloc( number_filter*sizeof(double*));
  double** bs1 = (double **)malloc( number_filter*sizeof(double*));
  biases_grad[0]=bv1;
  biases_grad[1]=bs1;
  double** bv2 = (double **)malloc( number_filter*sizeof(double*));
  double** bs2 = (double **)malloc( number_filter*sizeof(double*));
  biases_grad[3]=bv2;
  biases_grad[3]=bs2;
  double** bv3 = (double **)malloc( 128*sizeof(double*));
  double** bs3 = (double **)malloc( 128*sizeof(double*));
  biases_grad[4]=bv3;
  biases_grad[5]=bs3;
  double** bv4 = (double **)malloc( 2*sizeof(double*));
  double** bs4 = (double **)malloc( 2*sizeof(double*));
  biases_grad[6]=bv4;
  biases_grad[7]=bs4;
    */
  /*
  FIN PARAMETROS 
  */
  /*
  int itr;
  unsigned int epochs=10;
  unsigned int batch = 100;
  unsigned int batch_iterations= (tot_img_train_cat+tot_img_train_dog)/batch;
  double cost[epochs*batch];
  for(int i=0;i<epochs;i++){
    printf("%s %d\n","Epcoch: ",i);
    int vektor[tot_img_train_cat+tot_img_train_dog];
    int labels[batch];
    int contador=0;    
    random_batch(tot_img_train_cat+tot_img_train_dog, vektor,tot_img_train_cat+tot_img_train_dog);
    for(int j=0;j<batch_iterations;j++){
      
      double**** batch_images = (double****)malloc(batch*sizeof(double***));
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
      double loss_tot =optimizer(batch_images,batch, classes, lr,image_dim, image_channels, beta1, beta2, filters,weights,biases, filters_grad,weights_grad,biases_grad,itr,labels,number_filter2, conv_s, pool_s,f,pool_f,number_filter,depth_filter,dfilters,dbiases);
      cost[i*j]=loss_tot;
      free(batch_images);  
      
    
    }
  
  }*/

  double** b1_d = (double **)malloc( 5*sizeof(double*));
  matrix_initialize(5,1,b1_d);
  
  Matrix_t b1;
  b1.matrix=b1_d;
  b1.height=5;
  b1.width=1;
  

  return 0;
}
// gcc -o ejec multiplicarMatrices.c -lm 
// ./ejec aa -i asm name.bmp


//https://stackoverflow.com/questions/12747731/ways-to-create-dynamic-matrix-in-c







