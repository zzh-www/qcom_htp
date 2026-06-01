/* host validation: int16-internal solve -> int8 output.  Compare int8-output T to fp64, and to
 * directly quantizing the TRUE fp64 T to int8 — the op must not add error beyond that int8 floor. */
#include "src/gdn_solve_core.h"
#include <cstdio>
#include <cstdlib>
int main(){
    int32_t dims[2]; FILE*f=fopen("dims.i32","rb"); if(!f){perror("dims");return 1;}
    if(fread(dims,4,2,f)!=2)return 1; fclose(f);
    int H=dims[0],C=dims[1];
    int16_t*Aq=(int16_t*)malloc((size_t)H*C*C*2); float*sA=(float*)malloc((size_t)H*4);
    f=fopen("A_codes.i16","rb"); if(fread(Aq,2,(size_t)H*C*C,f)!=(size_t)H*C*C)return 1; fclose(f);
    f=fopen("sA.f32","rb"); if(fread(sA,4,H,f)!=(size_t)H)return 1; fclose(f);
    /* output int8 T at a per-head int8 scale (absmax(T)~1 -> sTout ~ 1/127); use a fixed 1.0/127 */
    float sTout = 1.0f/127.0f;
    int8_t*T=(int8_t*)malloc((size_t)H*C*C);
    for(int h=0;h<H;++h) gdn_solve_head_q<int8_t>(Aq+(size_t)h*C*C,C,sA[h],sTout,127.0f,T+(size_t)h*C*C);
    f=fopen("T_i8.f64","wb");
    for(size_t i=0;i<(size_t)H*C*C;++i){ double v=(double)T[i]*sTout; fwrite(&v,8,1,f);} fclose(f);
    printf("test_core: wrote T_i8.f64 (int8 out @ sTout=%.4e)\n",sTout);
    return 0;
}
