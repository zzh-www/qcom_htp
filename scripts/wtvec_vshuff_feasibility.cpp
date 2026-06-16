// Host-only (CPU) feasibility check for task (3): can the 64 vgather (gdn_pure_solve.cpp:263)
// be replaced by vshuff/vdeal? Replicates gp_hwlut_init with GP_CROUTON8=1 (production default),
// then for each output vec v=0..63 examines the 64 source halfword indices and asks whether
// they come from <= 2 aligned 128-byte (64-halfword) source vectors, and whether the in-vec
// permutation is a fixed structured shuffle (vshuff/vdeal family). Pure analysis, no device.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <string>

static uint16_t g_lut[4096];
static uint16_t g_hw[4096];

#define CROUTON_POS(r,c) ( (((r)&1)<<0) | (((c)&1)<<1) | ((((c)>>1)&1)<<2) | ((((c)>>2)&1)<<3) | \
    ((((c)>>3)&1)<<4) | ((((c)>>4)&1)<<5) | ((((r)>>1)&1)<<6) | ((((r)>>3)&1)<<7) | \
    ((((r)>>4)&1)<<8) | ((((r)>>5)&1)<<9) | ((((c)>>5)&1)<<10) | ((((r)>>2)&1)<<11) )

static void lut_init() {
    for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
        g_lut[r*64+c] = (uint16_t)CROUTON_POS(r,c);
}
static void hwlut_init() {
    int h = 0;
    for (int nt = 0; nt < 2; ++nt)
        for (int half = 0; half < 2; ++half)
            for (int kt = 0; kt < 2; ++kt)
                for (int grp = 0; grp < 8; ++grp)
                    for (int idx = 0; idx < 64; idx += 8)
                        for (int j = 0; j < 8; ++j) {
                            int vi = idx + j, lane = vi/16, off0 = (grp*8 + half*4 + lane)*16;
                            int off = off0 + (vi & 15), rgrp = off/128, rem = off%128;
                            int col = rem/4, row = rgrp*4 + rem%4;
                            g_hw[h++] = (uint16_t)(2 * g_lut[(kt*32+row)*64 + (nt*32+col)]);
                        }
}

int main() {
    lut_init();
    hwlut_init();
    // source = w_cv, 4096 int16 codes. g_hw[h] = byte offset = 2*halfword_idx.
    // each output vec v uses 64 offsets g_hw[v*64 .. v*64+63]; result lane l = src_hw[ g_hw[v*64+l]/2 ].
    // 128-byte HVX vec = 64 halfwords -> source vec id = src_hw_idx / 64, in-vec lane = src_hw_idx % 64.

    int n_one_src = 0, n_two_src = 0, n_many_src = 0;
    int n_vshuff_ok = 0;            // <=2 src AND in-vec perm is a structured shuffle/deal
    int n_identity = 0;            // single src, identity (pure copy)
    int n_perm_within_1 = 0;       // single src, lanes are a permutation of 0..63 (any perm)
    std::map<std::string,int> patterns;

    for (int v = 0; v < 64; ++v) {
        std::set<int> srcs;
        int lane_src[64], lane_in[64];
        for (int l = 0; l < 64; ++l) {
            int sh = g_hw[v*64+l] / 2;     // source halfword index
            int sv = sh / 64, si = sh % 64;
            lane_src[l] = sv; lane_in[l] = si;
            srcs.insert(sv);
        }
        int ns = (int)srcs.size();
        if (ns == 1) ++n_one_src; else if (ns == 2) ++n_two_src; else ++n_many_src;

        // contiguity of the (<=2) source vecs
        bool contiguous = true;
        if (ns == 2) { auto it = srcs.begin(); int a = *it; int b = *(++it); contiguous = (b == a+1); }

        // Build a signature of the gather as (which-src per lane, in-vec idx per lane).
        // For single-src: check identity / permutation.
        if (ns == 1) {
            bool ident = true, perm;
            std::set<int> seen;
            for (int l = 0; l < 64; ++l) { if (lane_in[l] != l) ident = false; seen.insert(lane_in[l]); }
            perm = (seen.size() == 64);
            if (ident) ++n_identity;
            if (perm) ++n_perm_within_1;
        }

        // vshuff/vdeal recognizer.
        // A 2-input vshuff(Vu,Vv,Rt) / vdeal interleaves two source vectors at a power-of-2 grain.
        // We test: for the (<=2 contiguous srcs) gather, does the lane->src mapping follow a single
        // grain-based interleave bit, and is the in-vec index a fixed function consistent with vshuff?
        // Conservative structural test: is there a single bit position p (0..5) such that
        //   src_select(l) == ((l >> p) & 1) flipped/notflipped consistently, AND
        //   in-vec idx == l with bit p moved to bit0 region (vshuff) -- we instead just record the
        //   full (src_select, in_idx) signature and count how many DISTINCT signatures exist; if a
        //   gather equals a vshuff its signature will match the canonical vshuff signature for some grain.
        // Simpler decisive criterion below.

        // Decisive vshuff/vdeal feasibility test:
        // vshuff/vdeal of 2 src vecs produces, at grain g (=2^p halfwords), an output where
        // output lane l selects src = bit, and reorders. Rather than enumerate all masks, we directly
        // test all 6 vshuff grains and all 6 vdeal grains against the actual lane mapping.
        bool matched = false;
        std::string mpat;
        // candidate masks: Rt for vshuff is element size in BYTES (here halfword ops). The HVX
        // 2-input shuffle Q6_W_vshuff_VVR(Vu,Vv,Rt) interleaves at byte grain Rt. We model the
        // halfword-level effect for grains 2,4,8,16,32,64,128 bytes (=1,2,4,8,16,32,64 halfwords).
        for (int gi = 0; gi < 6 && !matched; ++gi) {
            int grain = 1 << gi;   // in halfwords, 1..32
            // vshuff: out[2k]=A[k], out[2k+1]=B[k] at grain blocks. Model: split 64 lanes into
            // blocks of 'grain'; vshuff interleaves blocks of A(src0) and B(src1).
            // Build expected mapping for "vshuff(src1,src0,grain)" : even grain-block from src0, odd from src1.
            bool ok = true;
            for (int l = 0; l < 64 && ok; ++l) {
                int blk = l / grain;          // which grain-block in output
                int within = l % grain;
                int exp_src_is1 = blk & 1;    // odd output block -> from 2nd operand
                int exp_in = (blk/2)*grain + within;
                // map src0 to the lower source vec, src1 to higher
                auto it = srcs.begin(); int s0 = *it; int s1 = (ns==2)? *(++it) : s0;
                int want_src = exp_src_is1 ? s1 : s0;
                if (lane_src[l] != want_src || lane_in[l] != exp_in) ok = false;
            }
            if (ok) { matched = true; mpat = "vshuff_g" + std::to_string(grain); }
        }
        // vdeal: out[k]=A[2k], out[k+32]=A[2k+1] style (deinterleave). Test grains.
        for (int gi = 0; gi < 6 && !matched; ++gi) {
            int grain = 1 << gi;
            bool ok = true;
            for (int l = 0; l < 64 && ok; ++l) {
                int half = l / 32;            // first 32 lanes vs last 32
                int k = l % 32;
                int blk = k / grain, within = k % grain;
                int exp_in = (blk*2 + half)*grain + within;
                auto it = srcs.begin(); int s0 = *it;
                if (lane_src[l] != s0 || lane_in[l] != exp_in) ok = false;
            }
            if (ok) { matched = true; mpat = "vdeal_g" + std::to_string(grain); }
        }
        if (ns == 1) {
            // single-src: identity / rotation / vshuff-self
            bool ident = true;
            for (int l = 0; l < 64; ++l) if (lane_in[l] != l) ident = false;
            if (ident && !matched) { matched = true; mpat = "identity"; }
        }

        if (matched && (ns == 1 || (ns == 2 && contiguous))) { ++n_vshuff_ok; patterns[mpat]++; }
        else patterns["UNMATCHED(ns=" + std::to_string(ns) + (contiguous?")":",noncontig)")]++;
    }

    printf("=== wt-vec vgather -> vshuff/vdeal feasibility (GP_CROUTON8=1) ===\n");
    printf("source spread per output vec: 1-src=%d  2-src=%d  >2-src=%d  (of 64)\n", n_one_src, n_two_src, n_many_src);
    printf("single-src identity copies: %d ; single-src arbitrary perm: %d\n", n_identity, n_perm_within_1);
    printf("vshuff/vdeal decomposable (<=2 contig src, fixed structured mask): %d / 64\n", n_vshuff_ok);
    printf("pattern histogram:\n");
    for (auto &kv : patterns) printf("   %-28s %d\n", kv.first.c_str(), kv.second);
    return 0;
}
