const int IMAGE_H = 28;
const int IMAGE_W = 28;
const unsigned char IN_IMG[]  __attribute__((aligned(128))) ={
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x18, 0x20, 0x17, 
0xd, 0x8, 0x8, 0x8, 0x8, 0x8, 0xb, 0x10, 
0x12, 0xa, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x3d, 0x55, 0x3a, 0x19, 0x8, 0x8, 0x8, 
0x8, 0x9, 0x13, 0x1f, 0x26, 0xe, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x9, 0xc, 0x68, 0x8d, 0x50, 
0x20, 0x8, 0x8, 0x8, 0x8, 0x10, 0x48, 0x85, 
0x9e, 0x2c, 0xc, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0xa, 
0xf, 0x7c, 0xa2, 0x50, 0x1f, 0x8, 0x8, 0x8, 
0x9, 0x19, 0x86, 0xc4, 0xc3, 0x35, 0xd, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0xa, 0x12, 0x78, 0x97, 0x3b, 
0x18, 0x8, 0x8, 0x8, 0xc, 0x27, 0xca, 0xd6, 
0x91, 0x25, 0x9, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0xf, 
0x21, 0x9a, 0xb6, 0x37, 0x17, 0x8, 0x8, 0x8, 
0xf, 0x2d, 0xcc, 0xb9, 0x5a, 0x18, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x12, 0x31, 0xa1, 0xb1, 0x27, 
0x11, 0x8, 0x8, 0x8, 0x18, 0x44, 0xd1, 0xa0, 
0x2d, 0xf, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x16, 
0x3f, 0x93, 0x93, 0x13, 0xe, 0xe, 0xe, 0xc, 
0x2d, 0x6d, 0xd9, 0x8c, 0xa, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x1d, 0x5f, 0xb1, 0xba, 0x5b, 
0x5b, 0x5a, 0x53, 0x41, 0x71, 0xc5, 0xd9, 0x7b, 
0x9, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x1f, 
0x66, 0xbe, 0xdf, 0xb0, 0xb1, 0xb0, 0xaa, 0x9b, 
0xbb, 0xf0, 0xd7, 0x73, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x1b, 0x56, 0xb4, 0xed, 0xeb, 
0xeb, 0xea, 0xe9, 0xed, 0xf5, 0xf9, 0xcc, 0x69, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0xe, 
0x1e, 0x4f, 0x65, 0x50, 0x4f, 0x4a, 0x43, 0x62, 
0xa9, 0xf1, 0x89, 0x3a, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0xa, 0x1c, 0x22, 0x12, 
0x11, 0xe, 0xb, 0x53, 0xaa, 0xee, 0x69, 0x25, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0xd, 0x7d, 
0xcd, 0xed, 0x5e, 0x1d, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0xe, 0x83, 0xd1, 0xed, 0x5b, 0x1b, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0xd, 0x1b, 0x9a, 
0xe2, 0xec, 0x59, 0x1a, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x10, 0x25, 0xaa, 0xe7, 0xdd, 0x52, 0x17, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0xa, 0x12, 0x85, 
0xb2, 0xa2, 0x3a, 0x11, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0xb, 0x40, 0x55, 0x4c, 0x1d, 0xc, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 
0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 

};
