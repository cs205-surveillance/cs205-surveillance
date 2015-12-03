#define min(a, b) (((a) < (b)) ? (a) : (b))

inline float median9(float s0, float s1, float s2,
                     float s3, float s4, float s5,
                     float s6, float s7, float s8) {

    return min(s0, min(s1, min(s2, min(s3, min(s4, min(s5, min(s6, min(s7, s8))))))));
}
