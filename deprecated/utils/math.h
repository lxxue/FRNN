// #pragma once
// 
// struct int3 {
//     int x, y, z;
//     
//     int3() {}
//     int3(int x, int y, int z) : x(x), y(y), z(z) {}
//     int3 &operator= (const int3& op) { x=op.x; y=op.y; z=op.z; return *this; }
//     friend std::ostream& operator<<(std::ostream& out, const int3& op);
// };
// 
// struct float3 {
//     float x, y, z;
// 
//     float3() {}
//     float3(float x, float y, float z) : x(x), y(y), z(z) {}
//     float3(int3 op) : x(op.x), y(op.y), z(op.z) {}
// 
//     float3 &operator= (const float3& op) { x=op.x; y=op.y; z=op.z; return *this; }
//     float3 operator-(const float3& op) { return float3(x-op.x, y-op.y, z-op.z); }
//     float3 operator/(const float3& op) { return float3(x/op.x, y/op.y, z/op.z); }
//     friend std::ostream& operator<<(std::ostream& out, const float3& op);
// };

