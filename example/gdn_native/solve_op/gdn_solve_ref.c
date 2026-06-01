/* Scalar C reference for the GDN triangular-solve custom op: T = (I-A)^-1, A strictly-lower [C,C].
 * KDA-style: per-BL(=16) diagonal block forward substitution (int16 codes, int32 accumulation,
 * requant to int16 at each row write) + block-triangular merge.  Bit-faithful to the host golden
 * scripts/gdn_solve_int16_model.py — validate this on host BEFORE HVX-vectorizing (Phase C).
 *
 * Host test build:  cc -O2 gdn_solve_ref.c -o solve_ref -lm && ./solve_ref
 *   reads A_codes.i16 / sA.f32 / sT.f32 / dims.i32, writes T_out.f64, then compare in Python.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BL 16                              /* logical block (KDA sub-chunk BC) */

static int16_t clip16(double v){ if(v> 32767.) v= 32767.; if(v<-32767.) v=-32767.; return (int16_t)llround(v); }

/* one head: A int16 codes [C*C] (scale sA), -> T fp64 [C*C].  C must be a multiple of BL. */
static void solve_head(const int16_t *Aq, int C, double sA, double sT, double *T){
    int nb = C / BL;
    /* --- diagonal block inverses via forward substitution --- */
    /* Ti[b] = int16 codes (scale sT) of (I - A_bb)^-1 for the b-th BLxBL diagonal block */
    int16_t *Ti = (int16_t*)calloc((size_t)nb*BL*BL, sizeof(int16_t));
    for(int b=0; b<nb; ++b){
        int16_t *Tb = Ti + (size_t)b*BL*BL;     /* [BL,BL] */
        for(int i=0; i<BL; ++i){
            for(int j=0; j<BL; ++j){
                int32_t acc = 0;                /* int16*int16 -> int32, scale sA*sT */
                for(int k=0; k<i; ++k){
                    int16_t aik = Aq[(size_t)(b*BL+i)*C + (b*BL+k)];   /* A block (i,k), strictly-lower */
                    acc += (int32_t)aik * (int32_t)Tb[k*BL + j];
                }
                double ei = (j==i) ? 1.0/sT : 0.0;                    /* identity (exact) */
                Tb[i*BL + j] = clip16(ei + (double)acc * sA);         /* requant to int16 code @ sT */
            }
        }
    }
    /* --- block-triangular merge: T_ij = T_ii @ sum_{k=j..i-1} A_ik T_kj  (i>j) --- */
    /* keep block results in fp64 (dequantized), matching the golden model */
    double *Tblk = (double*)calloc((size_t)nb*nb*BL*BL, sizeof(double)); /* [nb][nb][BL][BL] */
    #define TBLK(i,j) (Tblk + (((size_t)(i)*nb + (j))*BL*BL))
    for(int b=0;b<nb;++b)
        for(int t=0;t<BL*BL;++t) TBLK(b,b)[t] = (double)Ti[(size_t)b*BL*BL + t] * sT;
    for(int i=0;i<nb;++i){
        for(int j=i-1;j>=0;--j){
            double acc[BL*BL]; for(int t=0;t<BL*BL;++t) acc[t]=0.0;
            for(int k=j;k<i;++k){
                for(int r=0;r<BL;++r) for(int c=0;c<BL;++c){
                    double s=0.0;
                    for(int m=0;m<BL;++m)
                        s += (double)Aq[(size_t)(i*BL+r)*C + (k*BL+m)] * sA * TBLK(k,j)[m*BL+c];
                    acc[r*BL+c]+=s;
                }
            }
            double *Tii = TBLK(i,i);
            for(int r=0;r<BL;++r) for(int c=0;c<BL;++c){
                double s=0.0; for(int m=0;m<BL;++m) s += Tii[r*BL+m]*acc[m*BL+c];
                TBLK(i,j)[r*BL+c]=s;
            }
        }
    }
    /* --- scatter blocks into full T --- */
    for(int t=0;t<C*C;++t) T[t]=0.0;
    for(int i=0;i<nb;++i) for(int j=0;j<=i;++j)
        for(int r=0;r<BL;++r) for(int c=0;c<BL;++c)
            T[(size_t)(i*BL+r)*C + (j*BL+c)] = TBLK(i,j)[r*BL+c];
    #undef TBLK
    free(Ti); free(Tblk);
}

int main(void){
    int32_t dims[2]; FILE*f;
    f=fopen("dims.i32","rb"); if(!f){perror("dims.i32");return 1;} fread(dims,4,2,f); fclose(f);
    int H=dims[0], C=dims[1];
    int16_t *Aq=malloc((size_t)H*C*C*2); float *sA=malloc((size_t)H*4); float sT;
    f=fopen("A_codes.i16","rb"); fread(Aq,2,(size_t)H*C*C,f); fclose(f);
    f=fopen("sA.f32","rb"); fread(sA,4,H,f); fclose(f);
    f=fopen("sT.f32","rb"); fread(&sT,4,1,f); fclose(f);
    double *T=malloc((size_t)H*C*C*8);
    for(int h=0;h<H;++h) solve_head(Aq+(size_t)h*C*C, C, sA[h], sT, T+(size_t)h*C*C);
    f=fopen("T_out.f64","wb"); fwrite(T,8,(size_t)H*C*C,f); fclose(f);
    printf("solve_ref: wrote T_out.f64 (H=%d C=%d)\n",H,C);
    return 0;
}
